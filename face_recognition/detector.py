"""
Face Detector Module
MTCNN-based face detection for real-time applications.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from mtcnn import MTCNN
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using MTCNN for high-accuracy detection."""
    
    def __init__(self, min_face_size: int = 80, confidence_threshold: float = 0.9):
        """
        Initialize the face detector.
        
        Args:
            min_face_size: Minimum face size in pixels
            confidence_threshold: Minimum confidence for detection
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self._model: Optional[MTCNN] = None
        self._initialized = False
    
    def load_model(self) -> None:
        """Load the MTCNN model."""
        if self._initialized:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info("Loading MTCNN model...")
            self._model = MTCNN()
            self._initialized = True
            logger.info("MTCNN model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MTCNN model: {e}")
            raise RuntimeError(f"Failed to load MTCNN model: {e}")
    
    def detect(self, image: np.ndarray, return_landmarks: bool = True) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (numpy array, BGR or RGB format)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of detected faces with bounding boxes and landmarks
            Each face dict contains:
                - 'box': [x, y, width, height]
                - 'confidence': detection confidence
                - 'landmarks': dict with 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'
                - 'quality': face quality score (0-1)
        """
        if not self._initialized:
            self.load_model()
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR format (OpenCV default)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Detect faces
        try:
            detections = self._model.detect_faces(rgb_image)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
        
        # Filter and process detections
        faces = []
        for detection in detections:
            # Extract bounding box
            x, y, width, height = detection['box']
            confidence = detection['confidence']
            
            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue
            
            # Filter by minimum face size
            if width < self.min_face_size or height < self.min_face_size:
                continue
            
            # Calculate face quality
            quality = self._calculate_face_quality(image, (x, y, width, height))
            
            # Prepare face data
            face_data = {
                'box': [x, y, width, height],
                'confidence': float(confidence),
                'quality': quality
            }
            
            # Add landmarks if requested
            if return_landmarks and 'keypoints' in detection:
                face_data['landmarks'] = {
                    'left_eye': detection['keypoints']['left_eye'],
                    'right_eye': detection['keypoints']['right_eye'],
                    'nose': detection['keypoints']['nose'],
                    'mouth_left': detection['keypoints']['mouth_left'],
                    'mouth_right': detection['keypoints']['mouth_right']
                }
            
            faces.append(face_data)
        
        logger.debug(f"Detected {len(faces)} faces")
        return faces
    
    def _calculate_face_quality(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate face quality score based on blur and brightness.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Quality score between 0 and 1
        """
        x, y, width, height = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)
        
        # Extract face region
        face_region = image[y:y2, x:x2]
        
        if face_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_region
        
        # Calculate blur score using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_quality = min(1.0, blur_score / 100.0)  # Normalize to 0-1
        
        # Calculate brightness score
        brightness = np.mean(gray)
        brightness_quality = 1.0 - abs(brightness - 127.5) / 127.5
        
        # Combine scores
        quality = (blur_quality * 0.7 + brightness_quality * 0.3)
        
        return float(quality)
    
    def extract_face(self, image: np.ndarray, bbox: List[int], 
                     target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """
        Extract and resize face from image.
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            target_size: Target size for extracted face (width, height)
            
        Returns:
            Extracted and resized face image, or None if extraction fails
        """
        x, y, width, height = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)
        
        # Extract face region
        face = image[y:y2, x:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
        
        return face_resized
    
    def align_face(self, image: np.ndarray, landmarks: Dict[str, Tuple[int, int]],
                   target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """
        Align face based on eye landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks dict
            target_size: Target size for aligned face (width, height)
            
        Returns:
            Aligned face image, or None if alignment fails
        """
        try:
            # Get eye positions
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # Calculate angle between eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            
            # Calculate center point between eyes
            eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                          (left_eye[1] + right_eye[1]) // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            
            # Rotate image
            h, w = image.shape[:2]
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
            
            # Extract and resize face
            # For now, just resize the whole rotated image
            aligned = cv2.resize(rotated, target_size, interpolation=cv2.INTER_LINEAR)
            
            return aligned
        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return None
    
    def __repr__(self) -> str:
        return f"FaceDetector(min_face_size={self.min_face_size}, confidence_threshold={self.confidence_threshold})"
