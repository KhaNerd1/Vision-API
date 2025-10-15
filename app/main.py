from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import time
import logging
from pathlib import Path
from PIL import Image

from app.models.detector import ObjectDetector
from app.schemas import DetectionResponse, Detection, BoundingBox, HealthResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VisionAPI - Object Detection Service",
    description="Production-ready object detection API using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allows requests from web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize detector (this happens once when the server starts)
logger.info("Initializing object detector...")
try:
    detector = ObjectDetector()
    logger.info("Object detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    detector = None

# Supported image formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image(file: UploadFile) -> None:
    """
    Validate uploaded file is an image
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file is not a valid image
    """
    # Check content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, BMP, or WebP)"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "VisionAPI - Object Detection Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "detect": "/api/v1/detect",
            "detect_annotated": "/api/v1/detect/annotated",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "description": "Upload images to detect objects using YOLOv8"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service health status and model information
    """
    return HealthResponse(
        status="healthy" if detector else "unhealthy",
        model="YOLOv8",
        model_loaded=detector is not None,
        version="1.0.0"
    )


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(
        0.5, 
        ge=0.1, 
        le=1.0, 
        description="Confidence threshold (0.1-1.0)"
    ),
    iou_threshold: float = Query(
        0.45,
        ge=0.1,
        le=1.0,
        description="IOU threshold for non-maximum suppression"
    )
):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (JPEG, PNG, BMP, WebP)
        confidence: Minimum confidence threshold for detections
        iou_threshold: IOU threshold for removing duplicate detections
        
    Returns:
        JSON with detected objects, bounding boxes, and confidence scores
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector not initialized. Service unavailable."
        )
    
    # Validate image
    validate_image(file)
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Start timing
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        file_extension = Path(file.filename).suffix
        temp_file = UPLOAD_DIR / f"{request_id}{file_extension}"
        
        # Read and save file
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        
        # Get image dimensions
        with Image.open(temp_file) as img:
            image_size = {"width": img.width, "height": img.height}
        
        # Run object detection
        logger.info(f"Processing request {request_id}: {file.filename}")
        detections = detector.detect_objects(
            str(temp_file),
            confidence=confidence,
            iou_threshold=iou_threshold
        )
        
        # Clean up temporary file
        os.remove(temp_file)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(
            f"Request {request_id} completed: {len(detections)} objects "
            f"detected in {processing_time:.3f}s"
        )
        
        # Format response
        return DetectionResponse(
            request_id=request_id,
            detections=[
                Detection(
                    class_name=d["class"],
                    confidence=d["confidence"],
                    bbox=BoundingBox(**d["bbox"])
                ) for d in detections
            ],
            count=len(detections),
            processing_time=round(processing_time, 3),
            image_size=image_size
        )
    
    except Exception as e:
        # Clean up on error
        if temp_file.exists():
            os.remove(temp_file)
        
        logger.error(f"Detection error for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/api/v1/detect/annotated", tags=["Detection"])
async def detect_and_annotate(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(0.5, ge=0.1, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.1, le=1.0)
):
    """
    Detect objects and return annotated image with bounding boxes
    
    Args:
        file: Image file (JPEG, PNG, BMP, WebP)
        confidence: Minimum confidence threshold
        iou_threshold: IOU threshold for NMS
        
    Returns:
        Image file with bounding boxes drawn around detected objects
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector not initialized. Service unavailable."
        )
    
    # Validate image
    validate_image(file)
    
    request_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        file_extension = Path(file.filename).suffix
        temp_file = UPLOAD_DIR / f"{request_id}{file_extension}"
        output_file = OUTPUT_DIR / f"{request_id}_annotated{file_extension}"
        
        # Read and save file
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        
        # Detect and annotate
        logger.info(f"Annotating image for request {request_id}")
        detector.annotate_image(
            str(temp_file),
            str(output_file),
            confidence=confidence,
            iou_threshold=iou_threshold
        )
        
        # Clean up input file
        os.remove(temp_file)
        
        # Return annotated image
        return FileResponse(
            output_file,
            media_type=f"image/{file_extension[1:]}",
            filename=f"detected_{file.filename}",
            headers={
                "X-Request-ID": request_id
            }
        )
    
    except Exception as e:
        # Clean up on error
        if temp_file.exists():
            os.remove(temp_file)
        if output_file.exists():
            os.remove(output_file)
        
        logger.error(f"Annotation error for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Annotation failed: {str(e)}"
        )


@app.get("/api/v1/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the detection model
    
    Returns:
        Model type, classes, and capabilities
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector not initialized"
        )
    
    return detector.get_model_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)