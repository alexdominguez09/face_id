"""
Recognition Pipeline Module
Orchestrates face detection, recognition, and tracking for real-time processing.
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
import logging
import time
from pathlib import Path

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .tracker import FaceTracker
from .database import FaceDatabase
from .config import Config
from .utils import draw_boxes, save_image

logger = logging.getLogger(__name__)


class RecognitionPipeline:
    """
    Main pipeline for real-time face recognition.
    Integrates detection, recognition, tracking, and database operations.
    """
    
    def __init__(self, 
                 detector: Optional[FaceDetector] = None,
                 recognizer: Optional[FaceRecognizer] = None,
                 tracker: Optional[FaceTracker] = None,
                 database: Optional[FaceDatabase] = None,
                 config: Optional[Config] = None):
        """
        Initialize the recognition pipeline.
        
        Args:
            detector: Face detector instance (created if None)
            recognizer: Face recognizer instance (created if None)
            tracker: Face tracker instance (created if None)
            database: Face database instance (created if None)
            config: Configuration instance (uses default if None)
        """
        self.config = config or Config()
        
        # Initialize components
        self.detector = detector or FaceDetector(
            min_face_size=self.config.MIN_FACE_SIZE,
            confidence_threshold=self.config.DETECTION_CONFIDENCE
        )
        
        self.recognizer = recognizer or FaceRecognizer(
            model_name=self.config.RECOGNITION_MODEL,
            similarity_threshold=self.config.SIMILARITY_THRESHOLD,
            use_gpu=self.config.USE_GPU,
            gpu_device=self.config.GPU_DEVICE
        )
        
        self.tracker = tracker or FaceTracker(
            max_age=self.config.MAX_TRACK_AGE,
            iou_threshold=self.config.IOU_THRESHOLD
        )
        
        self.database = database or FaceDatabase(
            db_path=str(self.config.DB_PATH)
        )
        
        # Processing state
        self.frame_count = 0
        self.processing_times = []
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            logger.info("Pipeline already initialized")
            return
        
        logger.info("Initializing recognition pipeline...")
        
        # Ensure directories exist
        self.config.ensure_directories()
        
        # Initialize database
        self.database.initialize()
        
        # Load models (lazy loading, but can preload here)
        logger.info("Models will be loaded on first use")
        
        self._initialized = True
        logger.info("Recognition pipeline initialized successfully")
    
    def process_frame(self, frame: np.ndarray, 
                     recognize: bool = True,
                     save_detections: bool = False,
                     output_dir: Optional[str] = None) -> Dict:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR format)
            recognize: Whether to perform recognition (vs detection only)
            save_detections: Whether to save detected face images
            output_dir: Directory to save face images
            
        Returns:
            Dict with detection and recognition results
        """
        start_time = time.time()
        
        if not self._initialized:
            self.initialize()
        
        # Step 1: Detect faces
        detections = self.detector.detect(frame, return_landmarks=True)
        
        # Step 2: Track faces
        tracks = self.tracker.update(detections)
        
        # Step 3: Recognize faces (if enabled and on recognition interval)
        recognition_results = []
        
        if recognize and len(tracks) > 0:
            # Only recognize on certain frames to improve performance
            should_recognize = (
                self.frame_count % self.config.RECOGNITION_INTERVAL == 0 or
                self.frame_count == 0
            )
            
            if should_recognize:
                recognition_results = self._recognize_faces(
                    frame, tracks, save_detections, output_dir
                )
        
        # Update frame count
        self.frame_count += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only last 100 processing times for average calculation
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        # Prepare results
        results = {
            'frame_count': self.frame_count,
            'detections': detections,
            'tracks': tracks,
            'recognition_results': recognition_results,
            'processing_time': processing_time,
            'fps': 1.0 / processing_time if processing_time > 0 else 0,
            'avg_fps': self._calculate_avg_fps()
        }
        
        return results
    
    def _recognize_faces(self, frame: np.ndarray, tracks: List[Dict],
                        save_detections: bool = False,
                        output_dir: Optional[str] = None) -> List[Dict]:
        """
        Recognize faces from tracks.
        
        Args:
            frame: Input frame
            tracks: List of face tracks
            save_detections: Whether to save face images
            output_dir: Directory to save face images
            
        Returns:
            List of recognition results
        """
        results = []
        
        for track in tracks:
            # Extract face from frame
            face_image = self.detector.extract_face(frame, track['box'])
            
            if face_image is None:
                continue
            
            # Generate embedding
            embedding = self.recognizer.encode(face_image)
            
            if embedding is None:
                continue
            
            # Search for matching face in database
            match_result = self.database.find_matching_face(
                embedding, 
                threshold=self.config.SIMILARITY_THRESHOLD
            )
            
            if match_result:
                face_data, similarity = match_result
                
                # Update last seen
                self.database.update_last_seen(face_data['id'])
                
                result = {
                    'track_id': track['track_id'],
                    'face_id': face_data['id'],
                    'name': face_data['name'],
                    'similarity': similarity,
                    'confidence': track['confidence'],
                    'box': track['box'],
                    'is_known': True
                }
                
                logger.debug(f"Recognized: {face_data['name']} (similarity: {similarity:.3f})")
            else:
                # Unknown face - could add to database or mark as unknown
                result = {
                    'track_id': track['track_id'],
                    'face_id': None,
                    'name': 'Unknown',
                    'similarity': 0.0,
                    'confidence': track['confidence'],
                    'box': track['box'],
                    'is_known': False
                }
                
                logger.debug(f"Unknown face (track_id: {track['track_id']})")
            
            # Save face image if requested
            if save_detections and output_dir:
                filename = f"face_{track['track_id']}_{self.frame_count}.jpg"
                filepath = Path(output_dir) / filename
                save_image(face_image, str(filepath))
                result['image_path'] = str(filepath)
            
            results.append(result)
        
        return results
    
    def add_known_face(self, name: str, image: np.ndarray,
                       metadata: Optional[Dict] = None) -> int:
        """
        Add a known face to the database.

        Args:
            name: Person name
            image: Face image (will be detected and encoded)
            metadata: Additional metadata

        Returns:
            Face ID in database
        """
        if not self._initialized:
            self.initialize()

        # Use InsightFace directly for detection and encoding
        # This is more reliable than using MTCNN + InsightFace separately
        try:
            # Ensure recognizer model is loaded
            if not self.recognizer._initialized:
                self.recognizer.load_model()

            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Use InsightFace to detect and encode in one step
            faces = self.recognizer._model.get(rgb_image)

            if len(faces) == 0:
                raise ValueError("No face detected in the provided image")

            if len(faces) > 1:
                logger.warning(f"Multiple faces detected, using the first one")

            # Get embedding from the first face
            face = faces[0]
            embedding = face.embedding

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

        except Exception as e:
            logger.error(f"Failed to detect/encode face: {e}")
            raise ValueError(f"Failed to generate face embedding: {e}")

        # Check if face already exists (duplicate detection)
        duplicate_threshold = getattr(self.config, 'DUPLICATE_THRESHOLD', 0.85)
        existing_face = self.database.face_exists(embedding, threshold=duplicate_threshold)

        if existing_face:
            similarity = self.database._cosine_similarity(embedding, existing_face['embedding'])
            raise ValueError(
                f"Face already exists in database!\n"
                f"Existing face: {existing_face['name']} (ID: {existing_face['id']})\n"
                f"Similarity: {similarity:.3f}\n"
                f"Each unique face should only be enrolled once. "
                f"If this is a different person, please use a clearer image."
            )

        # Add to database
        face_id = self.database.add_face(name, embedding, metadata)

        logger.info(f"Added known face: {name} (ID: {face_id})")

        return face_id
    
    def add_known_face_from_file(self, name: str, image_path: str,
                                  metadata: Optional[Dict] = None) -> int:
        """
        Add a known face from an image file.
        
        Args:
            name: Person name
            image_path: Path to image file
            metadata: Additional metadata
            
        Returns:
            Face ID in database
        """
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return self.add_known_face(name, image, metadata)
    
    def get_known_faces(self) -> List[Dict]:
        """
        Get list of all known faces.
        
        Returns:
            List of face data
        """
        if not self._initialized:
            self.initialize()
        
        return self.database.list_faces()
    
    def remove_known_face(self, face_id: int) -> bool:
        """
        Remove a known face from the database.
        
        Args:
            face_id: Face ID to remove
            
        Returns:
            True if removed, False otherwise
        """
        if not self._initialized:
            self.initialize()
        
        return self.database.delete_face(face_id)
    
    def reset_tracker(self) -> None:
        """Reset the face tracker."""
        self.tracker.reset()
        logger.info("Tracker reset")
    
    def get_stats(self) -> Dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dict with statistics
        """
        return {
            'frame_count': self.frame_count,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_fps': self._calculate_avg_fps(),
            'known_faces': self.database.get_face_count() if self._initialized else 0,
            'active_tracks': self.tracker.get_track_count()
        }
    
    def _calculate_avg_fps(self) -> float:
        """Calculate average FPS from processing times."""
        if not self.processing_times:
            return 0.0
        
        avg_time = np.mean(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def visualize_results(self, frame: np.ndarray, results: Dict,
                         show_confidence: bool = True,
                         show_fps: bool = True) -> np.ndarray:
        """
        Visualize recognition results on frame.
        
        Args:
            frame: Input frame
            results: Results from process_frame
            show_confidence: Whether to show confidence scores
            show_fps: Whether to show FPS
            
        Returns:
            Frame with visualizations
        """
        output = frame.copy()
        
        faces_drawn = False
        
        # First, draw recognized faces (with name and similarity)
        if results.get('recognition_results'):
            faces_to_draw = []
            
            for rec in results['recognition_results']:
                face_info = {
                    'box': rec['box'],
                    'name': rec['name']
                }
                
                if show_confidence:
                    face_info['name'] = f"{rec['name']} ({rec['similarity']:.2f})"
                
                faces_to_draw.append(face_info)
            
            if faces_to_draw:
                output = draw_boxes(output, faces_to_draw, color=(0, 255, 0))  # Green for known
                faces_drawn = True
        
        # Also draw detected faces even if not recognized
        if results.get('detections'):
            faces_to_draw = []
            
            for det in results['detections']:
                box = det.get('box')
                if box:
                    # Check if already drawn (avoid duplicates)
                    x, y, w, h = box
                    already_drawn = False
                    if results.get('recognition_results'):
                        for rec in results['recognition_results']:
                            rec_box = rec.get('box')
                            if rec_box and abs(rec_box[0] - x) < 10 and abs(rec_box[1] - y) < 10:
                                already_drawn = True
                                break
                    
                    if not already_drawn:
                        face_info = {
                            'box': box,
                            'name': f"Unknown ({det.get('confidence', 0):.2f})"
                        }
                        faces_to_draw.append(face_info)
            
            if faces_to_draw:
                output = draw_boxes(output, faces_to_draw, color=(0, 0, 255))  # Red for unknown
        
        # Draw FPS
        if show_fps:
            fps_text = f"FPS: {results.get('avg_fps', 0):.1f}"
            cv2.putText(output, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw face count
            total_faces = len(results.get('detections', []))
            if total_faces > 0:
                face_count_text = f"Faces: {total_faces}"
                cv2.putText(output, face_count_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output
    
    def __repr__(self) -> str:
        return (f"RecognitionPipeline(frame_count={self.frame_count}, "
                f"initialized={self._initialized})")
