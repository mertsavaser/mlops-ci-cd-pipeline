"""
Component/Integration tests for the model serving layer.
Tests the interaction between the FastAPI app and feature engineering logic.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.features import hash_feature

client = TestClient(app)


def test_predict_endpoint_success():
    """
    Test the /predict endpoint with a sample JSON payload.
    Verifies HTTP 200 response and that the response contains the hashed feature value.
    """
    # Sample payload
    payload = {"user_id": "test_user_123"}
    
    # Send POST request to /predict endpoint
    response = client.post("/predict", json=payload)
    
    # Verify HTTP 200 response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Parse response JSON
    response_data = response.json()
    
    # Verify response structure
    assert "user_id" in response_data, "Response should contain 'user_id' field"
    assert "hashed_feature" in response_data, "Response should contain 'hashed_feature' field"
    
    # Verify the user_id matches the input
    assert response_data["user_id"] == payload["user_id"]
    
    # Verify the hashed_feature is an integer
    assert isinstance(response_data["hashed_feature"], int)
    
    # Verify the hashed_feature value is correct by comparing with direct function call
    expected_hash = hash_feature(payload["user_id"])
    assert response_data["hashed_feature"] == expected_hash, \
        f"Hashed feature value {response_data['hashed_feature']} should match expected {expected_hash}"
    
    # Verify the hashed_feature is within valid bucket range [0, 999]
    assert 0 <= response_data["hashed_feature"] < 1000, \
        f"Hashed feature {response_data['hashed_feature']} should be in range [0, 999]"


def test_predict_endpoint_multiple_users():
    """
    Test the /predict endpoint with multiple different user IDs.
    Verifies the endpoint handles different inputs correctly.
    """
    test_cases = [
        "user_1",
        "user_2",
        "different_user",
        "user_999",
    ]
    
    for user_id in test_cases:
        payload = {"user_id": user_id}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["user_id"] == user_id
        assert response_data["hashed_feature"] == hash_feature(user_id)
        assert 0 <= response_data["hashed_feature"] < 1000


def test_predict_endpoint_empty_string():
    """
    Test the /predict endpoint with an empty string user_id.
    Verifies edge case handling.
    """
    payload = {"user_id": ""}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    response_data = response.json()
    
    assert response_data["user_id"] == ""
    assert isinstance(response_data["hashed_feature"], int)
    assert 0 <= response_data["hashed_feature"] < 1000

