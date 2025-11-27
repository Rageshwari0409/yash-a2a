"""
Pytest configuration and fixtures for test suite.
"""
import pytest
import warnings
from fastapi.testclient import TestClient
from app import app

# Suppress the asyncio deprecation warning from mangum
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mangum.protocols.lifespan")


@pytest.fixture
def client():
    """
    Fixture that provides a TestClient instance for the FastAPI app.
    """
    return TestClient(app)


@pytest.fixture
def lambda_event():
    """
    Fixture that provides a sample AWS Lambda event for testing.
    """
    return {
        "resource": "/",
        "path": "/",
        "httpMethod": "GET",
        "headers": {
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        "queryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        "requestContext": {
            "requestId": "test-request-id",
            "accountId": "123456789012",
            "stage": "test",
        },
        "body": None,
        "isBase64Encoded": False,
    }


@pytest.fixture
def lambda_context():
    """
    Fixture that provides a mock AWS Lambda context.
    """
    class LambdaContext:
        def __init__(self):
            self.function_name = "test-function"
            self.function_version = "$LATEST"
            self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
            self.memory_limit_in_mb = 128
            self.aws_request_id = "test-request-id"
            self.log_group_name = "/aws/lambda/test-function"
            self.log_stream_name = "2023/01/01/[$LATEST]test"
            
        def get_remaining_time_in_millis(self):
            return 30000
    
    return LambdaContext()
