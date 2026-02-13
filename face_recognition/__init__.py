"""
Face Recognition Module
Core components for face detection, recognition, and tracking.
"""

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .tracker import FaceTracker
from .database import FaceDatabase
from .config import Config

__all__ = [
    'FaceDetector',
    'FaceRecognizer',
    'FaceTracker',
    'FaceDatabase',
    'Config'
]
