from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import requests
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
 
# Configure logging to write INFO and above logs to the /tmp directory for AWS Lambda compatibility
log_file_path = os.getenv('LOG_FILE_PATH', '/tmp/mdoc_app.log')
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


mcp = FastMCP("MDoc")
 
class MeetingUploadRequest(BaseModel):
    """Request model for uploading and processing meeting recordings."""
    file_url: str= Field(..., description="HTTPS/S3 URL pointing to a video already in S3")
    client_name: str = Field(..., description="Name of the client")
    auth_file_path: Optional[str] = Field(None, description="Path to file containing auth token")


class MeetingUploadResponse(BaseModel):
    """Response model for meeting upload results."""
    success: bool = Field(..., description="Whether the operation was successful")
    session_guid: Optional[str] = Field(None, description="Session GUID for document generation")
    transcript: Optional[List[Dict[str, Any]]] = Field(None, description="Full transcript with timestamps")
    screenshots: Optional[List[Dict[str, Any]]] = Field(None, description="Screenshot metadata")
    video_path: Optional[str] = Field(None, description="Temporary path used during processing")
    error: Optional[str] = Field(None, description="Error message, if the operation failed")
    message: str = Field(..., description="A general message about the operation's outcome")


class DocumentGenerationRequest(BaseModel):
    """Request model for generating meeting summary documents."""
    doc_title: str = Field(..., description="Title for the document")
    session_guid: str = Field(..., description="Session GUID from upload endpoint")
    doc_format: str = Field(default="PDF", description="Output format: 'PDF', 'DOCX', or 'Both'")
    client_name: Optional[str] = Field(None, description="Optional - only needed if session_guid not provided")
    auth_file_path: Optional[str] = Field(None, description="Path to file containing auth token")


class DocumentGenerationResponse(BaseModel):
    """Response model for document generation results."""
    success: bool = Field(..., description="Whether the operation was successful")
    download_url: Optional[str] = Field(None, description="S3 presigned download URL for the generated file")
    error: Optional[str] = Field(None, description="Error message, if the operation failed")
    message: str = Field(..., description="A general message about the operation's outcome")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    success: bool = Field(..., description="Whether the service is healthy")
    message: str = Field(..., description="Health check status message")

    
@mcp.tool()
async def upload_meeting(request: MeetingUploadRequest) -> str:
    """Upload and process a meeting recording to extract transcript and screenshots.
 
    This tool sends a meeting video s3 url to the MDoc API for processing.
    It extracts transcripts, detects key moments, and captures screenshots based on the
    configured detection modes.
 
    Args:
        request: A MeetingUploadRequest object containing:
                 - file_url: HTTPS/S3 URL pointing to a video 
                 - client_name: Name of the client (required)
                 - auth_file_path: Optional path to authentication file

    Returns:
        A JSON string representing a MeetingUploadResponse object. This includes
        a 'success' boolean, 'session_guid' for document generation, 'transcript',
        'screenshots', and other processing metadata.
 
    Raises:
        Exception: Catches various exceptions during the HTTP request or response
                   processing and returns a structured error message.
    """
    try:
        url = f"{API_BASE_URL}/api/document/upload"
        headers = {}
        
        logging.info(f"Auth path: {request.auth_file_path}")

        # Get headers and metadata
        auth_headers, user_metadata = get_auth_headers_and_metadata(request.auth_file_path)
        headers.update(auth_headers)

        # Prepare form data
        data = {
            "client_name": request.client_name,
        }
        
        if request.file_url:
            data["file_url"] = request.file_url
            
        if user_metadata:
            data['user_metadata'] = json.dumps(user_metadata)
            
        logging.info(f"Attempting to upload meeting. Request URL: {url}, Data: {data}")
 
        response = requests.post(url, data=data, headers=headers, timeout=900)
        response.raise_for_status()
 
        response_data = response.json()
        logging.info(f"Meeting upload successful. Status: {response.status_code}")
 
        return MeetingUploadResponse(
            success=True,
            session_guid=response_data.get("session_guid"),
            transcript=response_data.get("transcript"),
            screenshots=response_data.get("screenshots"),
            video_path=response_data.get("video_path"),
            message="Meeting uploaded and processed successfully"
        ).model_dump_json()
 
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = e.response.text
        error_message = f"HTTP Error {status_code} from MDoc API: {error_detail}"
        logging.error(f"HTTPError in upload_meeting: {error_message}", exc_info=True)
        return MeetingUploadResponse(
            success=False,
            error=error_detail,
            message=error_message
        ).model_dump_json()
    except requests.exceptions.ConnectionError as e:
        error_message = f"Connection to MDoc API failed: {str(e)}. Please check the API server status."
        logging.error(f"ConnectionError in upload_meeting: {error_message}", exc_info=True)
        return MeetingUploadResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()
    except requests.exceptions.Timeout as e:
        error_message = f"MDoc API request timed out after 900 seconds: {str(e)}."
        logging.error(f"Timeout in upload_meeting: {error_message}", exc_info=True)
        return MeetingUploadResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()
    except requests.exceptions.RequestException as e:
        error_message = f"An unknown request error occurred with MDoc API: {str(e)}"
        logging.error(f"RequestException in upload_meeting: {error_message}", exc_info=True)
        return MeetingUploadResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()
    except Exception as e:
        error_message = f"An unexpected error occurred during meeting upload: {str(e)}"
        logging.error(f"Unexpected Exception in upload_meeting: {error_message}", exc_info=True)
        return MeetingUploadResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()


@mcp.tool()
async def generate_document(request: DocumentGenerationRequest) -> str:
    """Generate a meeting summary document from a processed video recording.
 
    This tool creates a professional meeting document (PDF, DOCX, or both) based on
    the transcript and screenshots from a previously uploaded meeting. The document
    can include process maps, missing questions, and screenshots.
 
    Args:
        request: A DocumentGenerationRequest object containing:
                 - doc_title: Title for the document (required)
                 - session_guid: Session GUID from upload endpoint (required)
                 - doc_format: Output format - "PDF", "DOCX", or "Both"
                 - client_name: Optional, only if session_guid not provided
                 - auth_file_path: Optional path to authentication file

    Returns:
        A JSON string representing a DocumentGenerationResponse object with
        a 'success' boolean, 'download_url' for the generated document, and a 'message'.
 
    Raises:
        Exception: Catches various exceptions during the HTTP request or response
                   processing and returns a structured error message.
    """
    try:
        url = f"{API_BASE_URL}/api/document/generate/user-story-generator"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        logging.info(f"Auth path: {request.auth_file_path}")

        # Get headers and metadata
        auth_headers, user_metadata = get_auth_headers_and_metadata(request.auth_file_path)
        headers.update(auth_headers)

        # Prepare form data
        data = {
            "doc_title": request.doc_title,
            "session_guid": request.session_guid,
            "doc_format": request.doc_format,
        }
        
        if request.client_name:
            data["client_name"] = request.client_name
        if user_metadata:
            data['user_metadata'] = json.dumps(user_metadata)
            
        logging.info(f"Attempting to generate document. Request URL: {url}, Data: {data}")
 
        response = requests.post(url, data=data, headers=headers, timeout=900)
        response.raise_for_status()
 
        response_data = response.json()
        logging.info(f"Document generation successful. Status: {response.status_code}")
 
        return DocumentGenerationResponse(
            success=True,
            download_url=response_data.get("download_url"),
            message="Document generated successfully"
        ).model_dump_json()
 
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = e.response.text
        error_message = f"HTTP Error {status_code} from MDoc API: {error_detail}"
        logging.error(f"HTTPError in generate_document: {error_message}", exc_info=True)
        return DocumentGenerationResponse(
            success=False,
            error=error_detail,
            message=error_message
        ).model_dump_json()
    except requests.exceptions.ConnectionError as e:
        error_message = f"Connection to MDoc API failed: {str(e)}. Please check the API server status."
        logging.error(f"ConnectionError in generate_document: {error_message}", exc_info=True)
        return DocumentGenerationResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()
    except requests.exceptions.Timeout as e:
        error_message = f"MDoc API request timed out after 900 seconds: {str(e)}."
        logging.error(f"Timeout in generate_document: {error_message}", exc_info=True)
        return DocumentGenerationResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()
    except requests.exceptions.RequestException as e:
        error_message = f"An unknown request error occurred with MDoc API: {str(e)}"
        logging.error(f"RequestException in generate_document: {error_message}", exc_info=True)
        return DocumentGenerationResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()
    except Exception as e:
        error_message = f"An unexpected error occurred during document generation: {str(e)}"
        logging.error(f"Unexpected Exception in generate_document: {error_message}", exc_info=True)
        return DocumentGenerationResponse(
            success=False,
            error=str(e),
            message=error_message
        ).model_dump_json()

 
if __name__ == "__main__":
    mcp.run()