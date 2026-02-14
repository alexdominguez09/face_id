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
    MIN_FACE_SIZE = 80
    DETECTION_CONFIDENCE = 0.9
    
    # Recognition settings
    RECOGNITION_MODEL = 'buffalo_l'
    SIMILARITY_THRESHOLD = 0.5
    EMBEDDING_DIM = 512
    
    # Tracking settings
    MAX_TRACK_AGE = 30
    IOU_THRESHOLD = 0.3
    
    # Processing settings
    RECOGNITION_INTERVAL = 5  # Recognize every N frames
    MAX_FACES = 10  # Maximum faces to process per frame
    
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
