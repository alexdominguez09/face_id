"""
Configuration Module
Centralized configuration for the face recognition system.
"""

import os
from pathlib import Path

class Config:
    """Configuration settings for face recognition."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    DB_PATH = DATA_DIR / 'faces.db'
    
    # Detection settings
    MIN_FACE_SIZE = 40  # Lowered for video surveillance with small faces
    DETECTION_CONFIDENCE = 0.8  # Slightly lower for better detection
    
    # Recognition settings
    RECOGNITION_MODEL = 'buffalo_l'
    SIMILARITY_THRESHOLD = 0.5
    EMBEDDING_DIM = 512
    DUPLICATE_THRESHOLD = 0.85  # Stricter threshold for detecting duplicate faces during enrollment
    
    # Tracking settings
    MAX_TRACK_AGE = 30
    IOU_THRESHOLD = 0.3
    
    # Processing settings
    RECOGNITION_INTERVAL = 5  # Recognize every N frames
    MAX_FACES = 10  # Maximum faces to process per frame
    
    # Face extraction settings
    FACE_EXTRACTION_PADDING = 0.0  # Percentage padding around bounding box (0.0-1.0)
    USE_FACE_ALIGNMENT = False  # Use eye-based face alignment when landmarks available
    USE_INSIGHTFACE_DETECTION = False  # Use InsightFace for detection instead of MTCNN (more accurate but slower)
    
    # Visualization settings
    SHOW_SIMILARITY_BARS = True  # Show similarity bars above bounding boxes
    SHOW_CONFIDENCE_SCORES = True  # Show confidence scores in labels
    
    # Logging settings
    ENABLE_CSV_LOGGING = True  # Enable CSV logging of recognition events
    CSV_LOG_MAX_SIZE_MB = 10  # Maximum CSV file size before rotation
    CSV_LOG_ROTATE = True  # Rotate log files when they get too large
    
    # Video output settings
    VIDEO_CODEC = 'mp4v'  # Codec for output videos: 'h264', 'h265', 'mp4v', 'xvid', 'avc1'
    VIDEO_QUALITY = 95  # Video quality (0-100, higher is better)
    VIDEO_CONTAINER = 'mp4'  # Container format: 'mp4', 'avi', 'mkv'
    
    # GPU settings
    USE_GPU = True  # Set to True if CUDA is properly installed
    GPU_DEVICE = 0
    
    # Web settings
    WEB_HOST = '0.0.0.0'
    WEB_PORT = 8000
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
