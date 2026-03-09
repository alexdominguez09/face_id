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
from .csv_logger import CSVLogger
from .config import Config
from .utils import draw_boxes, save_image, draw_similarity_bar, draw_metrics_overlay

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
        
        # CSV logger (optional)
        self.csv_logger = None
        if self.config.ENABLE_CSV_LOGGING:
            self.csv_logger = CSVLogger(
                max_file_size_mb=self.config.CSV_LOG_MAX_SIZE_MB,
                rotate_files=self.config.CSV_LOG_ROTATE
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
                     output_dir: Optional[str] = None,
                     source: str = 'frame_processing',
                     source_path: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR format)
            recognize: Whether to perform recognition (vs detection only)
            save_detections: Whether to save detected face images
            output_dir: Directory to save face images
            source: Source type for logging ('video', 'image', 'webcam', etc.)
            source_path: Path to source file for logging
            metadata: Additional metadata for logging
            
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
        
        # Step 3: Recognize faces (if enabled)
        recognition_results = []
        
        if recognize and len(tracks) > 0:
            # Always recognize on first frame, then every RECOGNITION_INTERVAL frames
            # This ensures we get immediate feedback
            should_recognize = (
                self.frame_count == 0 or
                self.frame_count % self.config.RECOGNITION_INTERVAL == 0
            )
            
            if should_recognize:
                recognition_results = self._recognize_faces(
                    frame, tracks, save_detections, output_dir
                )
            else:
                # On non-recognition frames, use last recognition results from tracks
                for track in tracks:
                    if 'last_recognition' in track:
                        recognition_results.append(track['last_recognition'])
            
            # DEBUG: Log recognition attempt
            logger.debug(f"Frame {self.frame_count}: recognize={recognize}, should_recognize={should_recognize}, results={len(recognition_results)}")
        
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
        
        # Log to CSV if enabled
        if self.csv_logger and (len(recognition_results) > 0 or len(detections) > 0):
            # Merge metadata
            log_metadata = {
                'processing_time': processing_time,
                'recognition_interval': self.config.RECOGNITION_INTERVAL
            }
            if metadata:
                log_metadata.update(metadata)
            
            self.csv_logger.log_recognition_results(
                frame_number=self.frame_count,
                results=results,
                source=source,
                source_path=source_path,
                metadata=log_metadata
            )
        
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
        
        # Convert frame to RGB for InsightFace if using InsightFace detection
        rgb_frame = None
        if self.config.USE_INSIGHTFACE_DETECTION:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get detections for this frame to access landmarks
        detections = self.detector.detect(frame, return_landmarks=True)
        
        # Create a mapping from bounding box to detection (for landmarks)
        detection_map = {}
        for det in detections:
            box = tuple(det['box'])
            detection_map[box] = det
        
        for track in tracks:
            track_box = tuple(track['box'])
            
            # Try to get landmarks from detection
            landmarks = None
            if track_box in detection_map:
                landmarks = detection_map[track_box].get('landmarks')
            
            # Method 1: Use InsightFace detection (more accurate, same as enrollment)
            if self.config.USE_INSIGHTFACE_DETECTION and rgb_frame is not None:
                embedding = self._recognize_with_insightface(rgb_frame, track['box'])
            # Method 2: Use MTCNN extraction + alignment (current method with improvements)
            else:
                embedding = self._recognize_with_mtcnn(frame, track['box'], landmarks)
            
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
            
            # Update track with recognition result
            self.tracker.update_track_recognition(track['track_id'], result)
            
            # Save face image if requested
            if save_detections and output_dir:
                # Extract face for saving
                face_image = self.detector.extract_face(
                    frame, 
                    track['box'],
                    padding_percent=self.config.FACE_EXTRACTION_PADDING
                )
                if face_image is not None:
                    filename = f"face_{track['track_id']}_{self.frame_count}.jpg"
                    filepath = Path(output_dir) / filename
                    save_image(face_image, str(filepath))
                    result['image_path'] = str(filepath)
            
            results.append(result)
        
        return results
    
    def _recognize_with_insightface(self, rgb_frame: np.ndarray, track_box: List[int]) -> Optional[np.ndarray]:
        """
        Recognize face using InsightFace detection (same as enrollment path).
        
        Args:
            rgb_frame: Input frame in RGB format
            track_box: Bounding box [x, y, width, height] from tracker
            
        Returns:
            Face embedding or None if recognition fails
        """
        try:
            # Ensure recognizer model is loaded
            if not self.recognizer._initialized:
                self.recognizer.load_model()
            
            # Use InsightFace to detect all faces in frame
            faces = self.recognizer._model.get(rgb_frame)
            
            if len(faces) == 0:
                logger.debug("No faces detected with InsightFace")
                return None
            
            # Find the face that best matches the tracker bounding box using IoU
            best_iou = 0
            best_face = None
            
            for face in faces:
                # InsightFace returns bbox as [x1, y1, x2, y2]
                iface_bbox = face.bbox.astype(int)
                iface_box = [iface_bbox[0], iface_bbox[1], 
                           iface_bbox[2] - iface_bbox[0], 
                           iface_bbox[3] - iface_bbox[1]]
                
                # Calculate IoU between tracker box and InsightFace box
                iou = self._calculate_iou(track_box, iface_box)
                
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_face = face
            
            if best_face is None:
                logger.debug(f"No InsightFace detection matches tracker box (best IoU: {best_iou:.2f})")
                return None
            
            # Use embedding from InsightFace (already normalized)
            embedding = best_face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"InsightFace recognition: IoU={best_iou:.2f}")
            return embedding
            
        except Exception as e:
            logger.error(f"InsightFace recognition failed: {e}")
            return None
    
    def _recognize_with_mtcnn(self, frame: np.ndarray, track_box: List[int], 
                             landmarks: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Recognize face using MTCNN extraction with optional alignment.
        
        Args:
            frame: Input frame in BGR format
            track_box: Bounding box [x, y, width, height]
            landmarks: Facial landmarks for alignment
            
        Returns:
            Face embedding or None if recognition fails
        """
        try:
            # Extract face with padding
            face_image = self.detector.extract_face(
                frame, 
                track_box,
                padding_percent=self.config.FACE_EXTRACTION_PADDING
            )
            
            if face_image is None:
                logger.debug("Face extraction failed")
                return None
            
            # Apply alignment if enabled and landmarks available
            if self.config.USE_FACE_ALIGNMENT and landmarks is not None:
                aligned_face = self.detector.align_face(frame, landmarks)
                if aligned_face is not None:
                    face_image = aligned_face
                    logger.debug("Applied face alignment")
            
            # Generate embedding
            embedding = self.recognizer.encode(face_image)
            
            return embedding
            
        except Exception as e:
            logger.error(f"MTCNN recognition failed: {e}")
            return None
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, w1, h1]
            box2: Second bounding box [x2, y2, w2, h2]
            
        Returns:
            IoU score (0.0-1.0)
        """
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
     
    def add_known_face(self, name: str, image: np.ndarray, 
                       metadata: Optional[Dict] = None) -> int:
        """
        Add a known face to the database.
        
        Args:
            name: Person name
            image: Face image (numpy array)
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

            # DEBUG: Log image properties
            logger.debug(f"add_known_face: image shape={image.shape if hasattr(image, 'shape') else 'no shape'}, dtype={image.dtype if hasattr(image, 'dtype') else 'no dtype'}")
            
            # Convert to RGB for InsightFace
            # Images from web are RGB, from CLI are BGR (cv2.imread)
            # We need to handle both cases
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Try to detect if image is BGR or RGB using same heuristic as above
                h, w = image.shape[:2]
                center_h, center_w = h // 2, w // 2
                crop_size = min(h, w) // 4
                y1 = max(0, center_h - crop_size)
                y2 = min(h, center_h + crop_size)
                x1 = max(0, center_w - crop_size)
                x2 = min(w, center_w + crop_size)
                
                if y2 > y1 and x2 > x1:
                    center_region = image[y1:y2, x1:x2]
                    avg_color = center_region.mean(axis=(0, 1))
                    
                    # In BGR: skin has B < R (blue < red)
                    # In RGB: skin has R > B (red > blue)
                    if avg_color[0] > avg_color[2]:  # B > R in BGR
                        # Likely BGR format, convert to RGB for InsightFace
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        logger.debug("Detected BGR image, converting to RGB for InsightFace")
                    else:
                        # Likely RGB format, use as-is
                        rgb_image = image
                        logger.debug("Detected RGB image, using as-is for InsightFace")
                else:
                    # Can't detect, assume BGR and convert to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.debug("Assuming BGR image, converting to RGB for InsightFace")
            else:
                rgb_image = image
                logger.debug(f"Using image as-is (not 3-channel color)")

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

        # Prepare metadata with image data
        import base64
        
        # Convert image to base64 for storage
        if metadata is None:
            metadata = {}
        
        # Store original image in metadata
        try:
            # Save image for web display
            # We need to ensure images are saved as RGB for proper web display
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Try to detect if image is BGR or RGB
                # Simple heuristic: check average color in center region (likely face)
                h, w = image.shape[:2]
                center_h, center_w = h // 2, w // 2
                crop_size = min(h, w) // 4
                y1 = max(0, center_h - crop_size)
                y2 = min(h, center_h + crop_size)
                x1 = max(0, center_w - crop_size)
                x2 = min(w, center_w + crop_size)
                
                if y2 > y1 and x2 > x1:
                    center_region = image[y1:y2, x1:x2]
                    # Average color of center region
                    avg_color = center_region.mean(axis=(0, 1))
                    
                    # In BGR: skin has B < R (blue < red)
                    # In RGB: skin has R > B (red > blue)
                    # Check if blue channel is larger than red (BGR format)
                    if avg_color[0] > avg_color[2]:  # B > R in BGR
                        # Likely BGR format, convert to RGB
                        image_to_save = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        logger.debug("Detected BGR image, converting to RGB for saving")
                    else:
                        # Likely RGB format, save as-is
                        image_to_save = image
                        logger.debug("Detected RGB image, saving as-is")
                else:
                    # Can't detect, assume BGR (from OpenCV) and convert to RGB
                    image_to_save = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.debug("Assuming BGR image, converting to RGB for saving")
            else:
                image_to_save = image
            
            # Encode image as JPEG
            success, encoded_image = cv2.imencode('.jpg', image_to_save)
            if success:
                image_bytes = encoded_image.tobytes()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                metadata['image_data'] = image_b64
                metadata['image_format'] = 'jpg'
                metadata['color_space'] = 'RGB'  # Always save as RGB for web
            else:
                logger.warning("Failed to encode image for storage in metadata")
        except Exception as e:
            logger.warning(f"Failed to store image in metadata: {e}")
            # Fallback: save original image
            try:
                success, encoded_image = cv2.imencode('.jpg', image)
                if success:
                    image_bytes = encoded_image.tobytes()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    metadata['image_data'] = image_b64
                    metadata['image_format'] = 'jpg'
            except:
                pass

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
        
        # Add image path to metadata
        if metadata is None:
            metadata = {}
        
        metadata['image_path'] = image_path
        
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
                         show_confidence: Optional[bool] = None,
                         show_fps: bool = True,
                         show_similarity_bars: Optional[bool] = None) -> np.ndarray:
        """
        Visualize recognition results on frame.
        
        Args:
            frame: Input frame
            results: Results from process_frame
            show_confidence: Whether to show confidence scores (None = use config)
            show_fps: Whether to show FPS
            show_similarity_bars: Whether to show similarity bars above bounding boxes (None = use config)
            
        Returns:
            Frame with visualizations
        """
        output = frame.copy()
        
        # Use config defaults if not specified
        if show_confidence is None:
            show_confidence = self.config.SHOW_CONFIDENCE_SCORES
        if show_similarity_bars is None:
            show_similarity_bars = self.config.SHOW_SIMILARITY_BARS
        
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
                
                # Draw similarity bar if enabled
                if show_similarity_bars and 'similarity' in rec:
                    output = draw_similarity_bar(output, rec['box'], rec['similarity'])
            
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
        
        # Draw metrics overlay
        if show_fps:
            metrics = {
                'fps': results.get('avg_fps', 0),
                'face_count': len(results.get('detections', [])),
                'known_count': len(results.get('recognition_results', []))
            }
            output = draw_metrics_overlay(output, metrics)
        
        return output
    
    def cleanup(self) -> None:
        """Clean up all components and release GPU memory."""
        logger.info("Cleaning up recognition pipeline...")
        
        # Clean up recognizer (releases GPU memory)
        if hasattr(self, 'recognizer') and self.recognizer:
            try:
                self.recognizer.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up recognizer: {e}")
        
        # Reset pipeline state
        self._initialized = False
        self.frame_count = 0
        self.processing_times = []
        
        logger.info("Recognition pipeline cleaned up")
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    def __repr__(self) -> str:
        return (f"RecognitionPipeline(frame_count={self.frame_count}, "
                f"initialized={self._initialized})")
