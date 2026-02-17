"""
Face Recognizer Module
InsightFace-based face recognition using ArcFace embeddings.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

# Try to import InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install with: pip install insightface")

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognizer using InsightFace ArcFace model."""
    
    def __init__(self, model_name: str = 'buffalo_l', similarity_threshold: float = 0.5, 
                 use_gpu: bool = True, gpu_device: int = 0):
        """
        Initialize the face recognizer.
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_m, buffalo_s, buffalo_sc)
            similarity_threshold: Threshold for face matching (cosine similarity)
            use_gpu: Whether to use GPU acceleration
            gpu_device: GPU device ID
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self._model = None
        self._initialized = False
        self.embedding_dim = 512  # ArcFace embedding dimension
    
    def load_model(self) -> None:
        """Load the InsightFace model."""
        if self._initialized:
            logger.info("Model already loaded")
            return
        
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "InsightFace is not installed. Install it with: pip install insightface"
            )
        
        try:
            logger.info(f"Loading InsightFace model: {self.model_name}")
            
            # Set context (GPU or CPU) - default to CPU if GPU fails
            ctx_id = self.gpu_device if self.use_gpu else -1
            
            # Initialize FaceAnalysis
            self._model = FaceAnalysis(
                name=self.model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Prepare model - try GPU first, fallback to CPU
            try:
                self._model.prepare(ctx_id=ctx_id, det_size=(640, 640))
            except Exception as gpu_err:
                # Fallback to CPU if GPU fails
                logger.warning(f"GPU loading failed ({gpu_err}), falling back to CPU")
                self._model.prepare(ctx_id=-1, det_size=(640, 640))
                self.use_gpu = False
            
            self._initialized = True
            logger.info(f"InsightFace model loaded successfully (GPU: {self.use_gpu})")
            
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise RuntimeError(f"Failed to load InsightFace model: {e}")
    
    def encode(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding.
        
        Args:
            face_image: Aligned face image (112x112 RGB)
            
        Returns:
            Face embedding vector (512-dimensional), or None if encoding fails
        """
        if not self._initialized:
            self.load_model()
        
        try:
            # Ensure image is RGB
            if len(face_image.shape) == 3:
                if face_image.shape[2] == 4:  # RGBA
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
                elif face_image.shape[2] == 3:
                    # Check if BGR (OpenCV default)
                    # Assume it's already RGB if coming from detector
                    pass
            
            # Get face embedding using InsightFace
            faces = self._model.get(face_image)
            
            if len(faces) == 0:
                logger.warning("No face detected in the image for encoding")
                return None
            
            # Return embedding of the first (and should be only) face
            embedding = faces[0].embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return None
    
    def encode_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple faces.
        
        Args:
            face_images: List of aligned face images
            
        Returns:
            List of face embedding vectors
        """
        embeddings = []
        for face_image in face_images:
            embedding = self.encode(face_image)
            embeddings.append(embedding)
        return embeddings
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First face embedding (normalized)
            embedding2: Second face embedding (normalized)
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        # Ensure embeddings are normalized
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to 0-1 range (cosine similarity is -1 to 1)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def compare_euclidean(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using Euclidean distance.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Distance score (lower is more similar)
        """
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(distance)
    
    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                 threshold: Optional[float] = None) -> bool:
        """
        Check if two embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            True if embeddings match, False otherwise
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        similarity = self.compare(embedding1, embedding2)
        return similarity >= threshold
    
    def find_best_match(self, query_embedding: np.ndarray, 
                        known_embeddings: List[np.ndarray],
                        threshold: Optional[float] = None) -> Tuple[Optional[int], float]:
        """
        Find the best matching embedding from a list.
        
        Args:
            query_embedding: Query face embedding
            known_embeddings: List of known face embeddings
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Tuple of (best_match_index, similarity_score) or (None, 0.0) if no match
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        if len(known_embeddings) == 0:
            return None, 0.0
        
        best_idx = None
        best_similarity = 0.0
        
        for idx, known_emb in enumerate(known_embeddings):
            similarity = self.compare(query_embedding, known_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        # Check if best match exceeds threshold
        if best_similarity >= threshold:
            return best_idx, best_similarity
        else:
            return None, best_similarity
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of face embeddings."""
        return self.embedding_dim
    
    def __repr__(self) -> str:
        return (f"FaceRecognizer(model_name='{self.model_name}', "
                f"similarity_threshold={self.similarity_threshold}, "
                f"use_gpu={self.use_gpu})")
