"""
AWS Lambda handler for Code Generation MCP Server.

Generic minimal entry point that wraps the main server for Lambda deployment.
"""
import os
import logging
import asyncio
from mangum import Mangum

# Configure logging for AWS Lambda
log_file_path = os.getenv('LOG_FILE_PATH', '/tmp/app.log')
logging.basicConfig(
    level=logging.INFO,
    filemode="a",
    filename=log_file_path,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)


def create_lambda_app():
    """Create FastAPI app for Lambda by importing server creation logic from main."""
    # Set up asyncio event loop for Lambda environment
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Import the server creation function from main
    from agents.sample_agent.main import create_server_instance
    
    # Get Lambda configuration from environment
    base_url = os.getenv('LAMBDA_URL', 'http://localhost:5000/')
    host = "0.0.0.0"
    port = int(os.getenv("SAMPLE_AGENT_PORT", 5000))
    
    # Create server using main's logic and return the app
    server = create_server_instance(host, port, base_url)
    return server.app


# Create the app and Lambda handler
app = create_lambda_app()
handler = Mangum(app)