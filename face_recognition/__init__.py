"""
Face Recognition Module
Core components for face detection, recognition, and tracking.
"""

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .tracker import FaceTracker
from .database import FaceDatabase
from .config import Config
from .pipeline import RecognitionPipeline
from .video_processor import VideoProcessor, RealTimeProcessor

__all__ = [
    'FaceDetector',
    'FaceRecognizer',
    'FaceTracker',
    'FaceDatabase',
    'Config',
    'RecognitionPipeline',
    'VideoProcessor',
    'RealTimeProcessor'
]
