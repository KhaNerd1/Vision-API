from pydantic import BaseModel, Field
from typing import List, Optional


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class Detection(BaseModel):
    """Single object detection result"""
    class_name: str = Field(..., alias="class", description="Detected object class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    
    class Config:
        populate_by_name = True


class DetectionResponse(BaseModel):
    """Response containing all detections"""
    request_id: str = Field(..., description="Unique request identifier")
    detections: List[Detection] = Field(..., description="List of detected objects")
    count: int = Field(..., description="Number of objects detected")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_size: Optional[dict] = Field(None, description="Original image dimensions")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str