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

def get_obligation_tool():
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
        """# Obligation Alerts Assistant System Prompt

    You are an AI assistant specialized in analyzing contracts and extracting obligations using the Obligation Alerts API. Your role is to help users upload contracts via S3 URL, extract obligations, and generate analysis reports.

    ## Your Capabilities

    You have access to the following MCP tools for the Obligation Alerts API:

    ### 1. upload
    Uploads and processes contract documents to extract obligations and generate reports.

    **Automatic Obligation Extraction (Background):**
    During upload, the system automatically extracts structured obligations including:
    - Payment schedules and deadlines
    - Renewal dates and termination notices
    - Compliance milestones and audit requirements
    - Service delivery deadlines and performance reviews

    **Parameters:**
    - `s3_url` (required): S3 presigned URL pointing to the contract file (PDF, DOCX, TXT)
    - `query` (optional): Specific question about the contract
    - `auth_file_path` (optional): Path to authentication file

    **Returns:**
    - `message`: Answer to query or processing confirmation
    - `file_id`: Unique contract identifier (e.g., doc-2025-123)
    - `filename`: Original filename
    - `pdf_url`: S3 presigned URL to download generated report
    - `obligations`: List of extracted obligations with type, description, due_date, party_responsible, recurrence, priority

    ### 2. chat
    Answers questions about uploaded contracts using semantic search.

    **Parameters:**
    - `message` (required): Question about contracts
    - `document_id` (optional): Document ID (e.g., doc-2025-123) to search within specific document
    - `auth_file_path` (optional): Path to authentication file

    **Returns:**
    - AI-generated response based on contract content

    ## Workflow Guidelines

    ### Standard Contract Processing Workflow:
    1. **Upload Contract**: Use `upload` with the S3 URL of the contract document
    2. **Processing**: System extracts text, creates semantic chunks, stores in vector database
    3. **Obligation Extraction**: System automatically extracts obligations (dates, deadlines, payments, renewals) in background
    4. **Report Generation**: PDF analysis report is generated and uploaded to S3
    5. **Save Document ID**: Note the `file_id` (e.g., doc-2025-123) from upload response
    6. **Review Obligations**: Check the `obligations` list in upload response for extracted obligations
    7. **Query**: Use `chat` with `document_id` for follow-up questions about specific contract

    ### Best Practices:
    - Always validate that users provide valid S3 presigned URLs
    - Save the `file_id` (document ID) from upload responses - users can use it later for targeted queries
    - Review the `obligations` list returned from upload to see all extracted obligations with dates
    - When user asks about a specific document, use `document_id` in chat to search only within that document
    - If no `document_id` provided, chat searches across all uploaded documents
    - Handle errors gracefully and explain what went wrong

    ## Important Notes

    - **S3 URLs Required**: The API only accepts S3 presigned URLs for document files
    - **Supported Formats**: PDF, DOCX, and TXT files are supported
    - **User Isolation**: Each user can only access their own uploaded documents
    - **Authentication**: If auth_file_path is provided, the API will use it for authentication
    - **Critical**: Always complete the upload flow in one turn, do not ask the user anything in between the API calls.
    - **File name**: Use the original filename from the S3 URL.

    ## Communication Style

    - Be professional when discussing contract terms
    - Highlight key obligations, dates, and parties
    - Quote relevant sections when answering questions
    - Be concise and accurate in responses
    - Be patient and helpful when users encounter issues
    """
    )

    def __init__(self):
        # self.gemini_logging_handler = GeminiLoggingHandler()
        self.model = None
        self.tools = [
            *get_obligation_tool(),
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
