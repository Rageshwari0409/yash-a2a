"""
Unit tests for AWS Lambda handler function.
"""
import pytest
import json
from lambda_function import handler


class TestLambdaHandler:
    """Test suite for Lambda handler function."""
    
    def test_handler_get_request(self, lambda_event, lambda_context):
        """Test Lambda handler with GET request."""
        response = handler(lambda_event, lambda_context)
        
        assert response["statusCode"] == 200
        assert "body" in response
        
        body = json.loads(response["body"])
        assert body == {"message": "Hello User"}
    
    def test_handler_response_headers(self, lambda_event, lambda_context):
        """Test Lambda handler response includes proper headers."""
        response = handler(lambda_event, lambda_context)
        
        assert "headers" in response
        assert response["statusCode"] == 200
    
    def test_handler_invalid_method(self, lambda_event, lambda_context):
        """Test Lambda handler with unsupported HTTP method."""
        lambda_event["httpMethod"] = "POST"
        
        response = handler(lambda_event, lambda_context)
        
        assert response["statusCode"] == 405
    
    def test_handler_invalid_path(self, lambda_event, lambda_context):
        """Test Lambda handler with invalid path."""
        lambda_event["path"] = "/invalid-path"
        
        response = handler(lambda_event, lambda_context)
        
        assert response["statusCode"] == 404
    
    def test_handler_with_query_params(self, lambda_event, lambda_context):
        """Test Lambda handler with query parameters."""
        lambda_event["queryStringParameters"] = {"test": "value"}
        
        response = handler(lambda_event, lambda_context)
        
        # Should still work for root path
        assert response["statusCode"] == 200
    
    def test_handler_response_is_json_serializable(self, lambda_event, lambda_context):
        """Test that Lambda handler response is JSON serializable."""
        response = handler(lambda_event, lambda_context)
        
        # Should not raise an exception
        json_response = json.dumps(response)
        assert json_response is not None
    
    def test_handler_preserves_request_context(self, lambda_event, lambda_context):
        """Test that handler can process events with request context."""
        response = handler(lambda_event, lambda_context)
        
        assert response is not None
        assert "statusCode" in response


class TestLambdaEventVariations:
    """Test suite for different Lambda event variations."""
    
    def test_handler_with_minimal_event(self, lambda_context):
        """Test handler with minimal event structure."""
        minimal_event = {
            "resource": "/",
            "path": "/",
            "httpMethod": "GET",
            "headers": {},
            "queryStringParameters": None,
            "pathParameters": None,
            "requestContext": {
                "requestId": "test-request-id",
                "accountId": "123456789012",
                "stage": "test",
            },
            "body": None,
            "isBase64Encoded": False,
        }
        
        response = handler(minimal_event, lambda_context)
        
        assert response["statusCode"] == 200
    
    def test_handler_with_base64_encoded_flag(self, lambda_event, lambda_context):
        """Test handler with base64 encoded flag set to True."""
        lambda_event["isBase64Encoded"] = True
        
        response = handler(lambda_event, lambda_context)
        
        # Should still process successfully
        assert "statusCode" in response
    
    def test_handler_different_stages(self, lambda_event, lambda_context):
        """Test handler with different API Gateway stages."""
        stages = ["dev", "staging", "prod"]
        
        for stage in stages:
            lambda_event["requestContext"]["stage"] = stage
            response = handler(lambda_event, lambda_context)
            
            assert response["statusCode"] == 200
