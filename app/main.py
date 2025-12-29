"""
Minimal FastAPI model serving application for MLOps homework.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from app.features import hash_feature

app = FastAPI(title="MLOps Homework Service", version="1.0.0")


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    user_id: str


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    user_id: str
    hashed_feature: int


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns HTTP 200 with a simple JSON message.
    """
    return {"status": "healthy", "message": "Service is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Prediction endpoint.
    Accepts a JSON payload with a user_id string field,
    applies feature hashing, and returns the hashed feature value.
    """
    hashed_value = hash_feature(request.user_id)
    
    return PredictResponse(
        user_id=request.user_id,
        hashed_feature=hashed_value
    )

