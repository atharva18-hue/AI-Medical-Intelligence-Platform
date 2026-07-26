from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict
    report: str
    gradcam_image: Optional[str] = None  # base64
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    device: str
