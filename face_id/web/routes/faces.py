"""
Face Management API Routes
Endpoints for enrolling, listing, viewing, and deleting faces.
"""

import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response, JSONResponse
from PIL import Image
import numpy as np

from face_recognition.database import FaceDatabase
from face_recognition.pipeline import RecognitionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces", tags=["faces"])

# Initialize components
_db: Optional[FaceDatabase] = None
_pipeline: Optional[RecognitionPipeline] = None


def get_database() -> FaceDatabase:
    """Get database instance."""
    global _db
    if _db is None:
        _db = FaceDatabase()
    return _db


def get_pipeline() -> RecognitionPipeline:
    """Get pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RecognitionPipeline()
    return _pipeline


@router.get("")
async def list_faces(limit: int = 100, offset: int = 0):
    """
    List all enrolled faces.
    """
    try:
        db = get_database()
        faces = db.list_faces(limit=limit, offset=offset)
        
        return {
            "faces": [
                {
                    "id": face["id"],
                    "name": face["name"],
                    "created_at": face.get("created_at"),
                    "updated_at": face.get("updated_at"),
                    "last_seen_at": face.get("last_seen_at"),
                    "seen_count": face.get("seen_count", 0),
                    "has_image": face.get("metadata", {}).get("image_path") is not None
                }
                for face in faces
            ],
            "total": db.get_face_count(),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{face_id}")
async def get_face(face_id: int):
    """
    Get details of a specific face.
    """
    try:
        db = get_database()
        face = db.get_face(face_id)
        
        if face is None:
            raise HTTPException(status_code=404, detail="Face not found")
        
        return {
            "id": face["id"],
            "name": face["name"],
            "embedding_dim": face.get("embedding_dim"),
            "created_at": face.get("created_at"),
            "updated_at": face.get("updated_at"),
            "last_seen_at": face.get("last_seen_at"),
            "seen_count": face.get("seen_count", 0),
            "metadata": face.get("metadata", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image/{face_id}")
async def get_face_image(face_id: int):
    """
    Get face image if available.
    """
    try:
        db = get_database()
        face = db.get_face(face_id)
        
        if face is None:
            raise HTTPException(status_code=404, detail="Face not found")
        
        metadata = face.get("metadata", {})
        image_data = metadata.get("image_data")
        
        if image_data:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            return Response(content=image_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="No image available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll")
async def enroll_face(request: Request):
    """
    Enroll a new face.
    
    Expects JSON body with:
        name: Person's name
        image: Base64 encoded image
        metadata: Optional JSON metadata
    """
    try:
        body = await request.json()
        
        # Get data from JSON request
        name = body.get('name')
        image = body.get('image')
        metadata = body.get('metadata')
        
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        if not image:
            raise HTTPException(status_code=400, detail="Image is required")
        
        # Decode base64 image
        if "," in image:
            # Handle data URL format
            image = image.split(",")[1]
        
        image_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(img)
        
        # Check for duplicates
        db = get_database()
        pipeline = get_pipeline()
        
        # Detect and encode face
        detections = pipeline.detector.detect(image_array)
        
        if not detections:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        if len(detections) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide an image with a single face")
        
        # Get face embedding - extract face then encode
        face_image = pipeline.detector.extract_face(image_array, detections[0]['box'])
        if face_image is None:
            raise HTTPException(status_code=400, detail="Could not extract face from image")
        
        embedding = pipeline.recognizer.encode(face_image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not encode face")
        
        # Check for duplicates
        existing = db.face_exists(embedding, threshold=0.85)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Face already exists: {existing['name']} (ID: {existing['id']}, similarity: high)"
            )
        
        # Add face to database
        import json
        meta = json.loads(metadata) if metadata else {}
        meta["image_data"] = image  # Store base64 image
        
        face_id = db.add_face(
            name=name,
            embedding=embedding,
            metadata=meta
        )
        
        logger.info(f"Enrolled face: {name} (ID: {face_id})")
        
        return {
            "face_id": face_id,
            "name": name,
            "message": "Face enrolled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/file")
async def enroll_face_file(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Enroll a face from an uploaded file.
    """
    try:
        # Read image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(img)
        
        # Check for duplicates
        db = get_database()
        pipeline = get_pipeline()
        
        # Detect and encode face
        detections = pipeline.detector.detect(image_array)
        
        if not detections:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        if len(detections) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected")
        
        # Get face embedding - extract face then encode
        face_image = pipeline.detector.extract_face(image_array, detections[0]['box'])
        if face_image is None:
            raise HTTPException(status_code=400, detail="Could not extract face from image")
        
        embedding = pipeline.recognizer.encode(face_image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not encode face")
        
        # Check for duplicates
        existing = db.face_exists(embedding, threshold=0.85)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Face already exists: {existing['name']} (ID: {existing['id']})"
            )
        
        # Add face to database
        import json
        import base64
        meta = {"image_data": base64.b64encode(image_bytes).decode()}
        
        face_id = db.add_face(
            name=name,
            embedding=embedding,
            metadata=meta
        )
        
        return {
            "face_id": face_id,
            "name": name,
            "message": "Face enrolled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{face_id}")
async def delete_face(face_id: int):
    """
    Delete a face from the database.
    """
    try:
        db = get_database()
        face = db.get_face(face_id)
        
        if face is None:
            raise HTTPException(status_code=404, detail="Face not found")
        
        deleted = db.delete_face(face_id)
        
        if deleted:
            logger.info(f"Deleted face ID: {face_id}")
            return {"message": "Face deleted successfully", "face_id": face_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete face")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_all_faces():
    """
    Delete all faces from the database.
    """
    try:
        db = get_database()
        count = db.get_face_count()
        
        # Get all faces and delete them
        faces = db.list_faces(limit=10000)
        deleted_count = 0
        
        for face in faces:
            if db.delete_face(face["id"]):
                deleted_count += 1
        
        logger.info(f"Cleared {deleted_count} faces from database")
        
        return {
            "message": f"Cleared {deleted_count} faces",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/query")
async def search_faces(name: str, exact: bool = False):
    """
    Search for faces by name.
    """
    try:
        db = get_database()
        
        if exact:
            face = db.get_face_by_name(name)
            if face:
                return {"faces": [face]}
            return {"faces": []}
        else:
            # Partial match
            all_faces = db.list_faces(limit=1000)
            matches = [f for f in all_faces if name.lower() in f["name"].lower()]
            
            return {"faces": matches}
            
    except Exception as e:
        logger.error(f"Error searching faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize")
async def recognize_face(image: str = Form(...)):
    """
    Recognize a face from an image.
    """
    try:
        # Decode base64 image
        if "," in image:
            image = image.split(",")[1]
        
        image_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(img)
        
        # Process frame
        pipeline = get_pipeline()
        result = pipeline.process_frame(image_array, recognize=True)
        
        return {
            "faces": result.get("faces", []),
            "detection_count": len(result.get("faces", [])),
            "recognized_count": len([f for f in result.get("faces", []) if f.get("name")])
        }
        
    except Exception as e:
        logger.error(f"Error recognizing face: {e}")
        raise HTTPException(status_code=500, detail=str(e))
