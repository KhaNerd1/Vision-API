from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object detection using YOLOv8
    
    This class loads a YOLO model and provides methods to:
    - Detect objects in images
    - Draw bounding boxes on images
    - Get detection results as structured data
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to the YOLO model file
            device: Device to run inference on ('cpu' or 'cuda' for GPU)
        """
        try:
            logger.info(f"Loading YOLO model from {model_path}")
            self.model = YOLO(model_path)
            self.device = device
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load YOLO model: {e}")
    
    def detect_objects(
        self, 
        image_path: str, 
        confidence: float = 0.5,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to the input image
            confidence: Minimum confidence threshold (0.0 to 1.0)
            iou_threshold: Intersection over Union threshold for NMS
            
        Returns:
            List of dictionaries containing detection results
            Each dict has: class, confidence, bbox (x1, y1, x2, y2)
        """
        try:
            logger.info(f"Running detection on {image_path}")
            
            # Run inference
            results = self.model(
                image_path,
                conf=confidence,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                # Extract each detected object
                for box in boxes:
                    detection = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x1": float(box.xyxy[0][0]),
                            "y1": float(box.xyxy[0][1]),
                            "x2": float(box.xyxy[0][2]),
                            "y2": float(box.xyxy[0][3])
                        }
                    }
                    detections.append(detection)
            
            logger.info(f"Found {len(detections)} objects")
            return detections
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise RuntimeError(f"Object detection failed: {e}")
    
    def annotate_image(
        self,
        image_path: str,
        output_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.45
    ) -> str:
        """
        Detect objects and draw bounding boxes on the image
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            confidence: Minimum confidence threshold
            iou_threshold: IOU threshold for NMS
            
        Returns:
            Path to the saved annotated image
        """
        try:
            logger.info(f"Annotating image: {image_path}")
            
            # Run inference
            results = self.model(
                image_path,
                conf=confidence,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # Get annotated image (with bounding boxes drawn)
            annotated_image = results[0].plot()
            
            # Save the annotated image
            cv2.imwrite(output_path, annotated_image)
            logger.info(f"Annotated image saved to {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Image annotation failed: {e}")
            raise RuntimeError(f"Failed to annotate image: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "YOLOv8",
            "task": self.model.task,
            "device": self.device,
            "classes": list(self.model.names.values())
        }