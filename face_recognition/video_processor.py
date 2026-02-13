"""
Video Processor Module
Handles video stream processing from cameras and video files.
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any
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
    
    def process_camera(self, camera_id: int = 0,
                      display: bool = True,
                      output_file: Optional[str] = None,
                      callback: Optional[Callable[[np.ndarray, Dict], None]] = None,
                      max_frames: Optional[int] = None) -> None:
        """
        Process video from camera in real-time.
        
        Args:
            camera_id: Camera device ID (default: 0)
            display: Whether to display video window
            output_file: Optional output video file path
            callback: Optional callback function for each frame
            max_frames: Maximum frames to process (None for unlimited)
        """
        logger.info(f"Starting camera processing (camera_id: {camera_id})")
        
        # Initialize pipeline
        self.pipeline.initialize()
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
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
                          skip_frames: int = 0) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            input_path: Input video file path
            output_path: Optional output video file path
            display: Whether to display video window
            callback: Optional callback function for each frame
            skip_frames: Number of frames to skip between processing (0 = process all)
            
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
                
                # Progress update
                if processed_count % 100 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            self._cleanup()
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_count,
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
