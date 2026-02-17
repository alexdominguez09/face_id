"""
Video Processor Module
Handles video stream processing from cameras and video files.
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, Union
import logging
import time
from pathlib import Path

from .pipeline import RecognitionPipeline
from .config import Config

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processor for real-time camera streams and video files.
    """
    
    def __init__(self, pipeline: Optional[RecognitionPipeline] = None,
                 config: Optional[Config] = None):
        """
        Initialize video processor.
        
        Args:
            pipeline: Recognition pipeline instance
            config: Configuration instance
        """
        self.config = config or Config()
        self.pipeline = pipeline or RecognitionPipeline(config=self.config)
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self._running = False
    
    def process_camera(self, camera_id: Union[int, str] = 0,
                      display: bool = True,
                      output_file: Optional[str] = None,
                      callback: Optional[Callable[[np.ndarray, Dict], None]] = None,
                      max_frames: Optional[int] = None,
                      save_faces_dir: Optional[str] = None,
                      show_fps: bool = True,
                      is_rtsp: bool = False) -> None:
        """
        Process video from camera in real-time.
        
        Args:
            camera_id: Camera device ID (default: 0), or RTSP URL string
            display: Whether to display video window
            output_file: Optional output video file path
            callback: Optional callback function for each frame
            max_frames: Maximum frames to process (None for unlimited)
            save_faces_dir: Directory to save detected face images
            show_fps: Whether to show FPS counter
            is_rtsp: Whether camera_id is an RTSP URL
        """
        logger.info(f"Starting camera processing (source: {camera_id}, is_rtsp: {is_rtsp})")
        
        # Initialize pipeline
        self.pipeline.initialize()
        
        # Open camera/stream
        if is_rtsp and isinstance(camera_id, str):
            # RTSP stream URL or HTTP stream
            logger.info(f"Opening stream: {camera_id}")
            # Try FFMPEG backend for network streams
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_FFMPEG)
            
            # Also try with OpenCV's default
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_id)
        else:
            # Regular camera - try multiple approaches
            camera_idx = int(camera_id)
            self.cap = None
            
            # Try different methods
            methods = [
                # Method 1: Direct V4L2 device path
                lambda: cv2.VideoCapture(f"/dev/video{camera_idx}", cv2.CAP_V4L2),
                # Method 2: V4L2 backend
                lambda: cv2.VideoCapture(camera_idx, cv2.CAP_V4L2),
                # Method 3: FFMPEG backend with device index
                lambda: cv2.VideoCapture(f"v4l2:///dev/video{camera_idx}", cv2.CAP_FFMPEG),
                # Method 4: Default backend
                lambda: cv2.VideoCapture(camera_idx),
                # Method 5: Try opening as file (some USB cameras appear as files)
                lambda: cv2.VideoCapture(f"/dev/video{camera_idx}"),
            ]
            
            for method in methods:
                try:
                    self.cap = method()
                    if self.cap is not None and self.cap.isOpened():
                        logger.info(f"Camera opened successfully")
                        break
                except Exception as e:
                    logger.debug(f"Method failed: {e}")
                    continue
            
            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError(
                    f"Failed to open camera {camera_id}.\n"
                    f"Possible solutions:\n"
                    f"  1. Check if camera is connected: ls -la /dev/video*\n"
                    f"  2. Try using a video file: --source /path/to/video.mp4\n"
                    f"  3. Try using RTSP stream: --source rtsp://camera-ip:554/stream\n"
                    f"  4. Check camera permissions: sudo chmod 666 /dev/video*"
                )
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        logger.info(f"Camera opened: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if output file specified
        if output_file:
            self._init_writer(output_file, width, height, fps)
        
        self._running = True
        frame_count = 0
        import time
        fps_time = time.time()
        fps_count = 0
        current_fps = 0
        
        # Create save directory if specified
        import os
        if save_faces_dir:
            os.makedirs(save_faces_dir, exist_ok=True)
        
        try:
            while self._running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame
                results = self.pipeline.process_frame(frame)
                
                # Visualize results
                output_frame = self.pipeline.visualize_results(frame, results)
                
                # Write to output file
                if self.writer:
                    self.writer.write(output_frame)
                
                # Display
                if display:
                    cv2.imshow('Face Recognition', output_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit requested by user")
                        break
                
                # Callback
                if callback:
                    callback(output_frame, results)
                
                # Calculate FPS
                fps_count += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_count
                    fps_count = 0
                    fps_time = time.time()
                
                # Show FPS on frame if enabled
                if show_fps and display:
                    cv2.putText(output_frame, f"FPS: {current_fps}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                frame_count += 1
                
                # Check max frames
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break
        
        finally:
            self._cleanup()
        
        logger.info(f"Processed {frame_count} frames")
    
    def process_video_file(self, input_path: str,
                          output_path: Optional[str] = None,
                          display: bool = False,
                          callback: Optional[Callable[[np.ndarray, Dict], None]] = None,
                          skip_frames: int = 0,
                          max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            input_path: Input video file path
            output_path: Optional output video file path
            display: Whether to display video window
            callback: Optional callback function for each frame
            skip_frames: Number of frames to skip between processing (0 = process all)
            max_frames: Maximum frames to process (None for all)
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing video file: {input_path}")
        
        # Initialize pipeline
        self.pipeline.initialize()
        
        # Open video file
        self.cap = cv2.VideoCapture(input_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_path}")
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Initialize video writer if output file specified
        if output_path:
            self._init_writer(output_path, width, height, fps)
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        total_faces_detected = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                results = self.pipeline.process_frame(frame)
                
                # Get face detections - pipeline returns 'detections' and 'tracks'
                detections = results.get('detections', [])
                tracks = results.get('tracks', [])
                
                # Total faces = detections from this frame
                if detections:
                    total_faces_detected += len(detections)
                    logger.debug(f"Frame {frame_count}: {len(detections)} face(s) detected")
                
                # Visualize results
                output_frame = self.pipeline.visualize_results(frame, results)
                
                # Write to output file
                if self.writer:
                    self.writer.write(output_frame)
                
                # Display
                if display:
                    cv2.imshow('Face Recognition', output_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit requested by user")
                        break
                
                # Callback
                if callback:
                    callback(output_frame, results)
                
                frame_count += 1
                processed_count += 1
                
                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    logger.info(f"Reached max frames limit: {max_frames}")
                    break
                
                # Progress update
                if processed_count % 100 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            self._cleanup()
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Video processing complete!")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Faces detected: {total_faces_detected}")
        logger.info(f"  Processing time: {elapsed_time:.1f}s")
        logger.info(f"  Average FPS: {processed_count / elapsed_time:.1f}")
        
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'faces_detected': total_faces_detected,
            'elapsed_time': elapsed_time,
            'avg_fps': processed_count / elapsed_time if elapsed_time > 0 else 0,
            'pipeline_stats': self.pipeline.get_stats()
        }
        
        logger.info(f"Processed {processed_count}/{frame_count} frames in {elapsed_time:.2f}s")
        logger.info(f"Average FPS: {stats['avg_fps']:.2f}")
        
        return stats
    
    def _init_writer(self, output_path: str, width: int, height: int, fps: int) -> None:
        """
        Initialize video writer.
        
        Args:
            output_path: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create writer
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        
        logger.info(f"Video writer initialized: {output_path}")
    
    def stop(self) -> None:
        """Stop processing."""
        self._running = False
        logger.info("Processing stopped")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        cv2.destroyAllWindows()
        
        logger.info("Resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()


class RealTimeProcessor:
    """
    Simplified real-time processor for quick usage.
    """
    
    def __init__(self, camera_id: int = 0, display: bool = True):
        """
        Initialize real-time processor.
        
        Args:
            camera_id: Camera device ID
            display: Whether to display video
        """
        self.camera_id = camera_id
        self.display = display
        self.processor = VideoProcessor()
    
    def run(self) -> None:
        """Run real-time processing."""
        self.processor.process_camera(
            camera_id=self.camera_id,
            display=self.display
        )
    
    def stop(self) -> None:
        """Stop processing."""
        self.processor.stop()
