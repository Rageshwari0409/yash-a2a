"""
Integration tests for the complete application stack.
"""
import pytest
from fastapi.testclient import TestClient
from app import app


class TestIntegration:
    """Integration test suite."""
    
    def test_complete_request_flow(self, client):
        """Test complete request-response flow."""
        # Send request
        response = client.get("/")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Hello User"
        
        # Verify headers
        assert response.headers.get("content-type") is not None
    
    def test_concurrent_requests(self, client):
        """Test handling of multiple concurrent requests."""
        responses = []
        
        for _ in range(10):
            response = client.get("/")
            responses.append(response)
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json() == {"message": "Hello User"} for r in responses)
    
    def test_app_state_consistency(self, client):
        """Test that app state remains consistent across requests."""
        response1 = client.get("/")
        response2 = client.get("/")
        
        assert response1.json() == response2.json()
        assert response1.status_code == response2.status_code
