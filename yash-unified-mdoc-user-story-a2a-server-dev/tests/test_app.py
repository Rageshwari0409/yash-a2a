"""
Unit tests for FastAPI application endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app import app


class TestGreetEndpoint:
    """Test suite for the greet endpoint."""
    
    def test_greet_success(self, client):
        """Test successful greeting response."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Hello User"}
    
    def test_greet_response_headers(self, client):
        """Test response headers are correctly set."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_greet_method_not_allowed(self, client):
        """Test that POST method is not allowed on greet endpoint."""
        response = client.post("/")
        
        assert response.status_code == 405
    
    def test_greet_put_method_not_allowed(self, client):
        """Test that PUT method is not allowed on greet endpoint."""
        response = client.put("/")
        
        assert response.status_code == 405
    
    def test_greet_delete_method_not_allowed(self, client):
        """Test that DELETE method is not allowed on greet endpoint."""
        response = client.delete("/")
        
        assert response.status_code == 405


class TestApplicationHealth:
    """Test suite for application health and metadata."""
    
    def test_openapi_schema_accessible(self, client):
        """Test that OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_accessible(self, client):
        """Test that API documentation is accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_invalid_route_404(self, client):
        """Test that invalid routes return 404."""
        response = client.get("/invalid-route")
        
        assert response.status_code == 404
    
    def test_invalid_route_with_path_params(self, client):
        """Test that invalid routes with path params return 404."""
        response = client.get("/invalid/route/123")
        
        assert response.status_code == 404
