"""
Video Processing API Routes
Endpoints for real-time video face detection and recognition.
"""

import base64
import io
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from PIL import Image
import numpy as np
import cv2

from face_recognition.pipeline import RecognitionPipeline
from face_recognition.config import Config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["video"])

# Global pipeline instance
_pipeline: Optional[RecognitionPipeline] = None

# FPS tracking
_frame_times = []


def get_pipeline() -> RecognitionPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RecognitionPipeline()
    return _pipeline


def calculate_fps() -> float:
    """Calculate current FPS."""
    global _frame_times
    now = time.time()
    
    # Remove old frames (older than 1 second)
    _frame_times = [t for t in _frame_times if now - t < 1.0]
    _frame_times.append(now)
    
    return len(_frame_times)


@router.post("/detect")
async def detect_faces(
    frame: str,
    recognize: bool = True
):
    """
    Detect and optionally recognize faces in a video frame.
    
    Args:
        frame: Base64 encoded image frame
        recognize: Whether to run recognition (default: True)
    """
    try:
        # Decode base64 frame
        if "," in frame:
            frame = frame.split(",")[1]
        
        image_bytes = base64.b64decode(frame)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Process frame
        pipeline = get_pipeline()
        result = pipeline.process_frame(image_array, recognize=recognize)
        
        # Calculate FPS
        fps = calculate_fps()
        
        # Format response
        faces = []
        for face_data in result.get("faces", []):
            face_info = {
                "bounding_box": face_data.get("bounding_box"),
                "confidence": face_data.get("confidence"),
                "name": face_data.get("name"),
                "face_id": face_data.get("face_id"),
                "similarity": face_data.get("similarity")
            }
            faces.append(face_info)
        
        return {
            "faces": faces,
            "face_count": len(faces),
            "recognized_count": len([f for f in faces if f.get("name")]),
            "fps": fps,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/start")
async def start_stream(source: str = "camera"):
    """
    Start a video stream for face detection.
    
    Args:
        source: Video source (camera, file, or rtsp URL)
    """
    try:
        # Initialize pipeline
        pipeline = get_pipeline()
        
        return {
            "status": "started",
            "source": source,
            "message": "Video stream started"
        }
        
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/stop")
async def stop_stream():
    """
    Stop the video stream.
    """
    global _frame_times
    _frame_times = []
    
    return {
        "status": "stopped",
        "message": "Video stream stopped"
    }


@router.post("/process/video")
async def process_video_file(
    input_path: str,
    output_path: Optional[str] = None,
    recognize: bool = True
):
    """
    Process a video file and save results.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video (optional)
        recognize: Whether to run recognition
    """
    try:
        import cv2
        
        pipeline = get_pipeline()
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output writer if path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            result = pipeline.process_frame(frame_rgb, recognize=recognize)
            
            # Draw results on frame
            for face_data in result.get("faces", []):
                box = face_data.get("bounding_box")
                if box:
                    x1, y1, x2, y2 = int(box["x"]), int(box["y"]), int(box["x2"]), int(box["y2"])
                    
                    # Draw box
                    color = (0, 255, 0) if face_data.get("name") else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = face_data.get("name", "Unknown")
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Store result
            results.append({
                "frame": frame_idx,
                "face_count": len(result.get("faces", []))
            })
            
            frame_idx += 1
            
            # Progress update every 100 frames
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        
        return {
            "status": "completed",
            "total_frames": frame_idx,
            "results": results,
            "output_path": output_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras")
async def list_cameras():
    """
    List available camera devices.
    """
    # Try to detect available cameras
    available = []
    
    try:
        import cv2
        
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append({
                    "index": i,
                    "name": f"Camera {i}"
                })
                cap.release()
                
    except Exception as e:
        logger.warning(f"Error detecting cameras: {e}")
    
    return {
        "cameras": available,
        "count": len(available)
    }


@router.get("/status")
async def get_stream_status():
    """
    Get current video stream status.
    """
    global _frame_times
    
    return {
        "active": len(_frame_times) > 0,
        "fps": calculate_fps(),
        "pipeline_ready": _pipeline is not None
    }
