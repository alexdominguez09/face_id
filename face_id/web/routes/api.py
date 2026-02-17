"""
General API Routes
Endpoints for system health, stats, settings, and utilities.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from face_recognition.database import FaceDatabase
from face_recognition.config import Config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["api"])

# Track start time for uptime
start_time = time.time()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/stats")
async def get_stats():
    """
    Get system statistics.
    """
    try:
        db = FaceDatabase()
        total_faces = db.get_face_count()
        
        # Get today's detections
        # For now, return placeholder - could track in database
        detections_today = 0
        
        return {
            "total_faces": total_faces,
            "detections_today": detections_today,
            "avg_processing_ms": 0,  # Could track this
            "uptime": get_uptime()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_faces": 0,
            "detections_today": 0,
            "avg_processing_ms": 0,
            "uptime": get_uptime()
        }


@router.get("/activity")
async def get_activity(limit: int = 20):
    """
    Get recent activity.
    """
    # This would typically come from a database table
    # For now, return empty list
    return {
        "activities": []
    }


@router.get("/settings")
async def get_settings():
    """
    Get system settings.
    """
    try:
        db = FaceDatabase()
        
        return {
            "total_faces": db.get_face_count(),
            "recognition_threshold": Config.SIMILARITY_THRESHOLD,
            "duplicate_threshold": Config.DUPLICATE_THRESHOLD,
            "database_path": str(Config.DB_PATH),
            "gpu_enabled": Config.USE_GPU
        }
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return {
            "total_faces": 0,
            "recognition_threshold": 0.5,
            "duplicate_threshold": 0.85
        }


@router.post("/settings")
async def update_settings(
    recognition_threshold: float = 0.5,
    duplicate_threshold: float = 0.85
):
    """
    Update system settings.
    """
    # Settings are typically read from Config
    # This could update a settings file
    return {
        "message": "Settings updated",
        "recognition_threshold": recognition_threshold,
        "duplicate_threshold": duplicate_threshold
    }


@router.get("/export")
async def export_database():
    """
    Export the face database as JSON.
    """
    try:
        db = FaceDatabase()
        faces = db.list_faces(limit=10000)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_faces": len(faces),
            "faces": [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "embedding_dim": f.get("embedding_dim"),
                    "created_at": f.get("created_at"),
                    "seen_count": f.get("seen_count", 0)
                }
                for f in faces
            ]
        }
        
        return export_data
        
    except Exception as e:
        logger.error(f"Error exporting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def import_database(data: dict):
    """
    Import faces from JSON data.
    """
    # This would require proper embedding handling
    raise HTTPException(status_code=501, detail="Import not yet implemented")


@router.get("/version")
async def get_version():
    """
    Get API version.
    """
    return {
        "version": "0.1.0",
        "api_name": "Face Recognition System",
        "build_date": "2025-02-14"
    }


@router.get("/models/status")
async def model_status():
    """
    Check if required models are loaded.
    """
    from face_recognition.pipeline import FaceRecognitionPipeline
    
    try:
        pipeline = FaceRecognitionPipeline()
        
        return {
            "detector_loaded": pipeline.detector is not None,
            "recognizer_loaded": pipeline.recognizer is not None,
            "tracker_loaded": pipeline.tracker is not None,
            "gpu_available": Config.USE_GPU
        }
    except Exception as e:
        return {
            "detector_loaded": False,
            "recognizer_loaded": False,
            "tracker_loaded": False,
            "gpu_available": False,
            "error": str(e)
        }


def get_uptime() -> str:
    """Get system uptime as string."""
    seconds = int(time.time() - start_time)
    
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"
