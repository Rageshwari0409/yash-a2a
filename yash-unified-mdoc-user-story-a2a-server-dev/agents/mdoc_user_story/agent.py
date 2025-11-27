from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from typing import Any, Dict, AsyncIterable, List, Literal, Optional
import os
import tempfile
from pydantic import BaseModel
import asyncio
import re
import json
from ..tool import mdoc_tools_client
import logging
# from utils.observability import GeminiLoggingHandler
from utils.obs import TokenTracker
from utils.config import get_model_config
from langchain_litellm import ChatLiteLLM

memory = MemorySaver()

def get_mdoc_tools():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    mdoc_tools = loop.run_until_complete(mdoc_tools_client.get_tools())

    return mdoc_tools

class ResponseFormat(BaseModel):
    """Respond to the user in this format.
        Args: 
            status: status of the request
            message: Text message to output from the LLM
            files: List of file paths that are provided by the Tools which are available to transfer, not included the files provided by the user
       """
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str
    files: List[str]

class MdocUserAgent:

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "application/zip"]
    SYSTEM_INSTRUCTION = (
        """# MDoc Assistant System Prompt

    You are an AI assistant specialized in helping users interact with the MDoc API - a Meeting Document Generator that transforms meeting recordings into professional documents. Your role is to guide users through the process of uploading meeting videos and generating comprehensive meeting summaries.

    ## Your Capabilities

    You have access to the following MCP tools for the MDoc API:

    ### 1. upload_meeting
    Uploads and processes meeting recordings to extract transcripts and screenshots.

    **Parameters:**
    - `file_url` (required): HTTPS/S3 URL pointing to the video file in S3
    - `client_name` (required): Name of the client
    - `auth_file_path` (optional): Path to authentication file

    **Returns:**
    - `session_guid`: Unique identifier for the session (use this for document generation)
    - `transcript`: Full transcript with timestamps
    - `screenshots`: Screenshot metadata with timestamps and reasons
    - `video_path`: Temporary processing path

    ### 2. generate_document
    Generates professional meeting summary documents (PDF, DOCX, or both) from processed recordings.

    **Parameters:**
    - `doc_title` (required): Title for the generated document
    - `session_guid` (required): Session GUID from the upload_meeting response
    - `doc_format` (optional): "PDF", "DOCX", or "Both" (defaults to "PDF")
    - `client_name` (optional): Only needed if session_guid not provided
    - `auth_file_path` (optional): Path to authentication file

    **Returns:**
    - `download_url`: S3 presigned URL to download the generated document
    - Success status and messages

    ## Workflow Guidelines

    ### Standard Meeting Processing Workflow:
    1. **Upload Meeting**: First use `upload_meeting` with the S3 video URL and client name
    2. **Wait for Processing**: The API will process the video and return a `session_guid`
    3. **Generate Document**: Use `generate_document` with the `session_guid` to create the final document
    4. **Download**: Provide the user with the download URL for their document

    ### Best Practices:
    - Always validate that users provide the required S3 URL for videos
    - Save the `session_guid` from upload responses as it's required for document generation
    - Handle errors gracefully and explain what went wrong

    ## Important Notes

    - **S3 URLs Required**: The API only accepts S3 URLs for video files, not direct file uploads
    - **Session Management**: Each upload creates a unique session_guid that must be used for document generation
    - **Authentication**: If auth_file_path is provided, the API will use it for authentication
    - **Format Options**: Documents can be generated as PDF, DOCX, or both formats.
    - **Critical**: Always complete the flow from upload to document generation in one turn do not ask the user anything in between the API calls.
    - **File name**: If file name is not provided by the user then name the file based on the client name.

    ## Communication Style

    - Be professional yet friendly
    - Provide clear step-by-step guidance
    - Set realistic expectations about processing times
    - Celebrate successful completions
    - Be patient and helpful when users encounter issues
    - Use the user's terminology (meeting, recording, video) naturally
    """
    )

    def __init__(self):
        # self.gemini_logging_handler = GeminiLoggingHandler()
        self.model = None
        self.tools = [
            *get_mdoc_tools(),
        ]
        logging.info(f"Tools:{self.tools}")

    def _extract_auth_token(self, query) -> Optional[str]:
        """Extract auth token from query structure."""
        if 'auth_token' in query:
            auth_token = query['auth_token'] 
        if 'user_metadata' in query:
            user_metadata = query['user_metadata']
            return auth_token, user_metadata
        return None

    async def invoke(self, query, sessionId) -> str:
        
        auth_token, user_metadata= self._extract_auth_token(query)
        auth_file_path = None
        team_id = user_metadata.get("team_id")

        async with get_model_config() as model_config_manager:
        
            # Get the team's model configuration
            team_config = await model_config_manager.get_team_model_config(team_id)
            model = team_config["selected_model"]
            provider = team_config["provider"]
            provider_model = f"{provider}/{model}"
            model_config = team_config["config"]
        
            # Create LLM instance with the team's configuration
            llm_params = {
                "model": provider_model,
                **model_config,
                "model_kwargs":{
                "thinking": {"type": "enabled", "budget_tokens": -1}
                }
            }
            print("LLM params:", llm_params)

            self.model = model
            self.team_id = team_id  # Store team_id for later use
            self.provider_model = provider_model  # Store provider_model for litellm call
            self.model_config = model_config  # Store model_config for litellm call
            self.graph = create_react_agent(model=ChatLiteLLM(**llm_params), tools=self.tools, checkpointer=memory, prompt= self.SYSTEM_INSTRUCTION)
        
        if auth_token:
            try:
                temp_fd, auth_file_path = tempfile.mkstemp(suffix='.json', prefix='auth_')
                with os.fdopen(temp_fd, 'w') as f:
                    auth_data = {
                        "auth_token": auth_token,
                        "user_metadata": user_metadata
                    }
                    json.dump(auth_data, f, indent=4)
                logging.info(f"Auth token and team id stored in temporary file: {auth_file_path}")
            except Exception as e:
                logging.error(f"Failed to create or write to auth temp file: {e}", exc_info=True)
                if 'temp_fd' in locals() and temp_fd is not None:
                    os.close(temp_fd)
                if auth_file_path and os.path.exists(auth_file_path):
                    os.remove(auth_file_path)
                auth_file_path = None

        file_paths_query = ""
        if query.get('files'):
            file_paths_query = "File Paths:\n" + "\n".join(
                f"- {file['uri']} ({file['mimeType']})" for file in query['files']
            )
        
        auth_info = f"\nAuth File Path: {auth_file_path}" if auth_file_path else ""
        
        full_query = query.get('text', '') + "\n\n" + file_paths_query + auth_info
        print("Full query", full_query)
        
        try:
            token_tracker = TokenTracker(model=self.model)
            langgraph_config = {"configurable": {"thread_id": sessionId}, "callbacks": [token_tracker]}
            agent_response = await self.graph.ainvoke({'messages': [('user', full_query)]}, langgraph_config)
                    
        except Exception as e:
            logging.critical(f"Exception during agent invocation: {e}", exc_info=True)
            # Clean up temp file on error
            if auth_file_path and os.path.exists(auth_file_path):
                os.remove(auth_file_path)
            # Re-raise the exception to be handled by the caller
            raise

        # Final cleanup of the temporary file
        if auth_file_path and os.path.exists(auth_file_path):
            try:
                os.remove(auth_file_path)
                logging.info(f"Successfully removed temporary auth file: {auth_file_path}")
            except OSError as e:
                logging.error(f"Error removing temporary auth file {auth_file_path}: {e}", exc_info=True)

        return await self.get_agent_response(langgraph_config)

    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the exchange rates...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the exchange rates..",
                }            
        
        yield self.get_agent_response(config)

        
    async def get_agent_response(self, langgraph_config):
        current_state = self.graph.get_state(langgraph_config)
        # Get messages from the state
        messages = current_state.values.get('messages', [])
        if messages:
            # Look for the ToolMessage containing the actual tool response
            tool_message_content = None
            
            # Iterate through messages in reverse to find the most recent ToolMessage
            for message in reversed(messages):
                if isinstance(message, ToolMessage):
                    tool_message_content = message.content
                    logging.info(f"Found ToolMessage content: {tool_message_content[:100]}...")
                    break
                elif isinstance(message, AIMessage):
                    content = message.content
                    if isinstance(content, list):
                        tool_message_content = "\n".join(str(item) for item in content)
                    elif not isinstance(content, str):
                        tool_message_content = str(content)
                    else:
                        tool_message_content = content
                    logging.info(f"Found ToolMessage content: {tool_message_content[:100]}...")
                    break

        logging.info(f"Number of messages found: {len(messages)}")

        if tool_message_content:
                
                # Use the same graph with token tracking for formatting
                try:
                    # Prepare the formatting prompt
                    formatting_prompt = f"""Format the following agent response according to the ResponseFormat schema:

                    Agent Response:
                    {tool_message_content}

                    Return a JSON object with:
                    - status: "input_required" | "completed" | "error"
                    - message: The formatted response text
                    - files: List of file paths mentioned in the response (empty list if none)

                    Analyze the response and determine the appropriate status based on:
                    - "completed": Task is finished successfully
                    - "input_required": More information needed from user
                    - "error": An error occurred

                    Return ONLY the JSON object, no additional text."""

                    # Reuse the same graph with the existing langgraph_config (includes token tracker)
                    formatting_response = await self.graph.ainvoke(
                        {'messages': [('user', formatting_prompt)]},
                        langgraph_config
                    )
                    
                    # Extract the AI's response from the graph output
                    ai_message = formatting_response['messages'][-1]
                    formatted_content = ai_message.content

                    if formatted_content.startswith("```"):
                    
                        pattern = r'```(?:json)?\s*\n(.*?)\n```'
                        match = re.search(pattern, formatted_content, re.DOTALL)
                        
                        if match:
                            formatted_content =  match.group(1).strip()
                        
                        # If no code block found, return original text
                        formatted_content =  formatted_content.strip()
                    
                    # Parse the formatted response
                    parsed_response = json.loads(formatted_content)
                    
                    return {
                        "is_task_complete": parsed_response.get("status") == "completed",
                        "require_user_input": parsed_response.get("status") == "input_required",
                        "content": parsed_response.get("message", tool_message_content),
                        "files": parsed_response.get("files", [])
                    }
                    
                except Exception as e:
                    logging.error(f"Error formatting response with graph: {e}", exc_info=True)
                    # Fallback to original response
                    return {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": tool_message_content
                    }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }
