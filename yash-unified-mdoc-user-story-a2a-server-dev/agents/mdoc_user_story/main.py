import os
import logging
import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from common.server import A2AServer
from common.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    MissingAPIKeyError,
    AgentConstraints
)
from common.utils.push_notification_auth import PushNotificationSenderAuth
from agents.mdoc_user_story.task_manager import AgentTaskManager
from agents.mdoc_user_story.agent import MdocUserAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filemode="a",
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_server_instance(host: str, port: int, base_url: str) -> A2AServer:
    """
    Create and configure the A2A server instance.
    
    This function is used by both main() and lambda_function.py to create the server.
    
    Args:
        host (str): Host address to bind the server
        port (int): Port number to run the server
        base_url (str): Base URL for the agent service
        
    Returns:
        A2AServer: Configured server instance
    """
    # Define agent capabilities
        # Define agent capabilities (no streaming or push notifications)
    capabilities = AgentCapabilities(
        streaming=False,
        pushNotifications=False
    )

    # Construct the agent card with metadata, skills, and security schema
    agent_card = AgentCard(
        name="M-DOC USER STORY GENERATION AGENT",
        description="This is a Meeting Document Generator Agent that specializes in transforming meeting recordings into professional, comprehensive documents with transcripts, screenshots, and structured summaries",
        url=f"http://{os.getenv('HOST')}:{port}/",
        version="1.0.0",
        defaultInputModes=MdocUserAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=MdocUserAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        agent_constraints=AgentConstraints(
                max_file_size="10MB",
                supported_file_types=[
                    "video/mp4",
                    "video/mov",
                    "video/avi",
                    "video/mkv"
                ],
                max_files=1,
                prompt_template=[
                    "Process my meeting recording for client {{client_name}} and generate a document titled {{doc_title}}",
                    "Upload and document the meeting for {{client_name}}",
                    "Generate a {{doc_format}} meeting summary for the recording "
                ],
                prompt_template_variable_name=["client_name", "doc_title", "doc_format"]
            ),
        skills=[
            AgentSkill(
                id="end-to-end-meeting-documentation",
                name="End-to-End Meeting Documentation",
                description=(
                    "Complete workflow automation that processes meeting recordings and generates professional "
                    "documents in a single operation. Combines video upload, transcript extraction, screenshot "
                    "capture, and document generation into one seamless process."
                ),
                tags=["Workflow Automation", "Meeting Documentation", "Full Pipeline", "One-Click Processing", "MCP"],
                examples=[
                    "Process and document my meeting for Acme Corp titled 'Product Launch Discussion'",
                    "Complete documentation workflow for the video provided, client name TechStart, title 'Q1 Business Review'",
                    "Upload, process, and generate PDF for Global Solutions - 'Employee Onboarding Session'",
                ],
            )
        ],
        customAgentMetaData="MDOC_USER_STORY_AGENT",
        agentType="SYNCHRONOUS_AGENT"
    )
    
    # Set up push notification sender authentication and generate JWK
    notification_sender_auth = PushNotificationSenderAuth()
    notification_sender_auth.generate_jwk()

    # Initialize the server with the agent card and task manager
    server = A2AServer(
        agent_card=agent_card,
        task_manager=AgentTaskManager(
            agent=MdocUserAgent(),
            notification_sender_auth=notification_sender_auth
        ),
        host=host,
        port=port,
    )

    # Add endpoint for serving the JWKs (for push notification verification)
    server.app.add_route(
        "/.well-known/jwks.json",
        notification_sender_auth.handle_jwks_endpoint,
        methods=["GET"],
    )

    logger.info(f"Server configured for {host}:{port}")
    return server


@click.command()
@click.option("--host", "host", default="0.0.0.0", help="Host address to bind the server.")
@click.option("--port", "port", default=int(os.getenv("PORT", 5000)))
def main(host, port):
    """
    Entry point for starting the Mdoc user story Agent server.

    This function sets up the agent card, authentication, and server,
    then starts the server to listen for incoming requests.
    """
    try:
        # Construct base URL
        base_url = f"http://{os.getenv('HOST')}:{port}/"
        
        # Create server using the shared function
        server = create_server_instance(host, port, base_url)

        logger.info(f"Starting mdoc user story on {host}:{port}")
        server.start()

    except MissingAPIKeyError as e:
        logger.error(f"Missing API Key Error: {e}")
        exit(1)
    except Exception as e:
        import traceback
        logger.error(f"An error occurred during server startup: {traceback.format_exc()}")
        exit(1)


if __name__ == "__main__":
    main()
