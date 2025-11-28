from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional, Dict
import requests
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
log_file_path = os.getenv('LOG_FILE_PATH', '/tmp/obligation_app.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_path, filemode="a")

API_BASE_URL = os.getenv('BACKEND_URL', "http://localhost:8000")

def get_auth_headers_and_metadata(auth_file_path: Optional[str]) -> tuple[Dict[str, str], Optional[Dict]]:
    """Helper function to get authorization headers and user metadata"""
    headers = {}
    user_metadata = None
    logging.info(f"Reading auth info from file: {auth_file_path}")
    if auth_file_path and os.path.exists(auth_file_path):
        try:
            with open(auth_file_path, 'r') as f:
                auth_data = json.load(f)

            auth_token = auth_data.get('auth_token')
            user_metadata = auth_data.get('user_metadata')

            logging.info(f"User metadata: {user_metadata}")

            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                logging.info(f"Using auth token from file: {len(auth_token)} characters")
            else:
                logging.warning("No auth token available")

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in auth file: {e}")
        except Exception as e:
            logging.error(f"Error reading auth info from file: {e}")
    else:
        logging.warning("Auth file path not provided or does not exist")

    return headers, user_metadata


mcp = FastMCP("ObligationAlerts")

class UploadRequest(BaseModel):
    """Request model for uploading and processing contract documents."""
    s3_url: str = Field(..., description="S3 presigned URL pointing to the contract document (PDF, DOCX, TXT)")
    query: Optional[str] = Field(None, description="Optional question to answer about the contract")
    auth_file_path: Optional[str] = Field(None, description="Path to file containing auth token")


class UploadResponse(BaseModel):
    """Response model for contract upload results."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Answer to query or processing confirmation")
    file_id: Optional[str] = Field(None, description="Unique contract identifier")
    filename: Optional[str] = Field(None, description="Original filename")
    pdf_url: Optional[str] = Field(None, description="S3 presigned URL to download generated report")
    error: Optional[str] = Field(None, description="Error message, if the operation failed")


class ChatRequest(BaseModel):
    """Request model for chatting with uploaded contracts."""
    message: str = Field(..., description="Question about the contracts")
    auth_file_path: Optional[str] = Field(None, description="Path to file containing auth token")


class ChatResponse(BaseModel):
    """Response model for chat results."""
    success: bool = Field(..., description="Whether the operation was successful")
    response: Optional[str] = Field(None, description="AI-generated response based on contract content")
    error: Optional[str] = Field(None, description="Error message, if the operation failed")
    message: str = Field(..., description="A general message about the operation's outcome")


@mcp.tool()
async def upload(request: UploadRequest) -> str:
    """Upload and process a contract document to extract obligations and generate report.

    This tool sends a contract document S3 URL to the Obligation Alerts API for processing.
    It extracts text, creates semantic chunks, stores in vector database, and generates PDF report.

    Args:
        request: An UploadRequest object containing:
                 - s3_url: S3 presigned URL pointing to contract file (PDF, DOCX, TXT)
                 - query: Optional question to answer about the contract
                 - auth_file_path: Optional path to authentication file

    Returns:
        A JSON string representing an UploadResponse object with success status,
        message (answer or confirmation), file_id, filename, and pdf_url.
    """
    try:
        url = f"{API_BASE_URL}/api/v1/upload"
        headers = {}

        logging.info(f"Auth path: {request.auth_file_path}")

        # Get headers and metadata
        auth_headers, user_metadata = get_auth_headers_and_metadata(request.auth_file_path)
        headers.update(auth_headers)

        # Prepare form data
        data = {
            "s3_url": request.s3_url,
            "user_metadata": json.dumps(user_metadata) if user_metadata else "{}",
        }

        if request.query:
            data["query"] = request.query

        logging.info(f"Attempting to upload contract. Request URL: {url}, Data: {data}")

        response = requests.post(url, data=data, headers=headers, timeout=900)
        response.raise_for_status()

        response_data = response.json()
        logging.info(f"Contract upload successful. Status: {response.status_code}")

        return UploadResponse(
            success=True,
            message=response_data.get("message"),
            file_id=response_data.get("file_id"),
            filename=response_data.get("filename"),
            pdf_url=response_data.get("pdf_url")
        ).model_dump_json()

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = e.response.text
        error_message = f"HTTP Error {status_code}: {error_detail}"
        logging.error(f"HTTPError in upload: {error_message}", exc_info=True)
        return UploadResponse(
            success=False,
            message=error_message,
            error=error_detail
        ).model_dump_json()
    except requests.exceptions.ConnectionError as e:
        error_message = f"Connection failed: {str(e)}"
        logging.error(f"ConnectionError in upload: {error_message}", exc_info=True)
        return UploadResponse(
            success=False,
            message=error_message,
            error=str(e)
        ).model_dump_json()
    except requests.exceptions.Timeout as e:
        error_message = f"Request timed out: {str(e)}"
        logging.error(f"Timeout in upload: {error_message}", exc_info=True)
        return UploadResponse(
            success=False,
            message=error_message,
            error=str(e)
        ).model_dump_json()
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logging.error(f"Exception in upload: {error_message}", exc_info=True)
        return UploadResponse(
            success=False,
            message=error_message,
            error=str(e)
        ).model_dump_json()


@mcp.tool()
async def chat(request: ChatRequest) -> str:
    """Chat with uploaded contracts using semantic search.

    This tool answers questions about previously uploaded contracts by searching
    stored document chunks and generating AI responses.

    Args:
        request: A ChatRequest object containing:
                 - message: Question about the contracts
                 - auth_file_path: Optional path to authentication file

    Returns:
        A JSON string representing a ChatResponse object with success status
        and AI-generated response based on contract content.
    """
    try:
        url = f"{API_BASE_URL}/api/v1/chat"
        headers = {}

        logging.info(f"Auth path: {request.auth_file_path}")

        # Get headers and metadata
        auth_headers, user_metadata = get_auth_headers_and_metadata(request.auth_file_path)
        headers.update(auth_headers)

        # Prepare form data
        data = {
            "message": request.message,
            "user_metadata": json.dumps(user_metadata) if user_metadata else "{}",
        }

        logging.info(f"Attempting to chat. Request URL: {url}, Data: {data}")

        response = requests.post(url, data=data, headers=headers, timeout=300)
        response.raise_for_status()

        response_data = response.json()
        logging.info(f"Chat successful. Status: {response.status_code}")

        return ChatResponse(
            success=True,
            response=response_data.get("response"),
            message="Chat completed successfully"
        ).model_dump_json()

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = e.response.text
        error_message = f"HTTP Error {status_code}: {error_detail}"
        logging.error(f"HTTPError in chat: {error_message}", exc_info=True)
        return ChatResponse(
            success=False,
            message=error_message,
            error=error_detail
        ).model_dump_json()
    except requests.exceptions.ConnectionError as e:
        error_message = f"Connection failed: {str(e)}"
        logging.error(f"ConnectionError in chat: {error_message}", exc_info=True)
        return ChatResponse(
            success=False,
            message=error_message,
            error=str(e)
        ).model_dump_json()
    except requests.exceptions.Timeout as e:
        error_message = f"Request timed out: {str(e)}"
        logging.error(f"Timeout in chat: {error_message}", exc_info=True)
        return ChatResponse(
            success=False,
            message=error_message,
            error=str(e)
        ).model_dump_json()
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logging.error(f"Exception in chat: {error_message}", exc_info=True)
        return ChatResponse(
            success=False,
            message=error_message,
            error=str(e)
        ).model_dump_json()


if __name__ == "__main__":
    mcp.run()