"""
Utility Functions
Helper functions for image processing and calculations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = False) -> np.ndarray:
    """
    Preprocess image for face detection/recognition.
    
    Args:
        image: Input image (BGR or RGB)
        target_size: Target size (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image
    """
    # Make a copy to avoid modifying original
    processed = image.copy()
    
    # Convert BGR to RGB if needed (OpenCV loads as BGR)
    if len(processed.shape) == 3 and processed.shape[2] == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if target_size:
        processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize if requested
    if normalize:
        processed = processed.astype(np.float32) / 255.0
    
    return processed


def align_face(image: np.ndarray, landmarks: Dict[str, Tuple[int, int]],
               target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
    """
    Align face based on eye landmarks.
    
    Args:
        image: Input image
        landmarks: Facial landmarks dict with 'left_eye' and 'right_eye'
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
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                      (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Rotate image
        h, w = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Extract face region around eyes
        # Calculate desired eye positions for target size
        desired_left_eye = (0.35, 0.4)  # Relative position in target image
        desired_right_eye_x = 1.0 - desired_left_eye[0]
        
        # Calculate scale
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - desired_left_eye[0]) * target_size[0]
        scale = desired_dist / dist if dist > 0 else 1.0
        
        # Update rotation matrix with scale
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Rotate and scale
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Extract face region
        # Calculate translation to move eyes to desired position
        tx = target_size[0] * 0.5 - eyes_center[0]
        ty = target_size[1] * desired_left_eye[1] - eyes_center[1]
        
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply final transformation
        aligned = cv2.warpAffine(image, M, target_size, flags=cv2.INTER_CUBIC)
        
        return aligned
        
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        return None


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize embeddings
    emb1_norm = embedding1 / np.linalg.norm(embedding1)
    emb2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Convert to 0-1 range (cosine similarity is -1 to 1)
    similarity = (similarity + 1) / 2
    
    return float(similarity)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Distance (lower is more similar)
    """
    return float(np.linalg.norm(embedding1 - embedding2))


def draw_boxes(image: np.ndarray, faces: List[Dict], 
               color: Tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        faces: List of face dicts with 'box' key
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with boxes drawn
    """
    output = image.copy()
    
    for face in faces:
        x, y, w, h = face['box']
        
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label if available
        if 'name' in face:
            label = face['name']
        elif 'track_id' in face:
            label = f"ID: {face['track_id']}"
        else:
            label = None
        
        if label:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                output,
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                output,
                label,
                (x, y - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
    
    return output


def draw_landmarks(image: np.ndarray, landmarks: Dict[str, Tuple[int, int]],
                   color: Tuple[int, int, int] = (0, 0, 255),
                   radius: int = 2) -> np.ndarray:
    """
    Draw facial landmarks on image.
    
    Args:
        image: Input image
        landmarks: Facial landmarks dict
        color: Landmark color (BGR)
        radius: Circle radius
        
    Returns:
        Image with landmarks drawn
    """
    output = image.copy()
    
    for point_name, point in landmarks.items():
        cv2.circle(output, point, radius, color, -1)
    
    return output


def resize_with_aspect_ratio(image: np.ndarray, width: Optional[int] = None,
                            height: Optional[int] = None,
                            inter: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        inter: Interpolation method
        
    Returns:
        Resized image
    """
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)


def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        path: File path
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if needed
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(path, image, params)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        path: File path
        
    Returns:
        Image array or None if loading fails
    """
    try:
        image = cv2.imread(path)
        
        if image is None:
            logger.error(f"Failed to load image: {path}")
            return None
        
        return image
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


def get_image_quality(image: np.ndarray) -> float:
    """
    Calculate image quality score based on blur.
    
    Args:
        image: Input image
        
    Returns:
        Quality score (0-1, higher is better)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance (blur measure)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize to 0-1 range
    quality = min(1.0, blur_score / 100.0)
    
    return float(quality)
