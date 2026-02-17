"""
Test Recognition Pipeline
End-to-end tests for the recognition pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from pathlib import Path
import logging

from face_recognition import (
    RecognitionPipeline,
    VideoProcessor,
    Config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline_initialization():
    """Test pipeline initialization."""
    logger.info("\n" + "="*50)
    logger.info("Testing Pipeline Initialization")
    logger.info("="*50)
    
    try:
        config = Config()
        pipeline = RecognitionPipeline(config=config)
        
        logger.info(f"✓ Pipeline created: {pipeline}")
        
        # Initialize
        pipeline.initialize()
        logger.info("✓ Pipeline initialized")
        
        # Check stats
        stats = pipeline.get_stats()
        logger.info(f"✓ Stats retrieved: {stats}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_add_known_face():
    """Test adding a known face."""
    logger.info("\n" + "="*50)
    logger.info("Testing Add Known Face")
    logger.info("="*50)
    
    try:
        config = Config()
        config.DB_PATH = Path('data/test_pipeline.db')
        
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Create a dummy face image (would normally load a real image)
        dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Draw a simple face-like pattern
        cv2.circle(dummy_image, (40, 40), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(dummy_image, (72, 40), 15, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(dummy_image, (56, 70), (20, 10), 0, 0, 180, (255, 255, 255), -1)  # Mouth
        
        # Add face
        face_id = pipeline.add_known_face("Test Person", dummy_image)
        logger.info(f"✓ Face added with ID: {face_id}")
        
        # List faces
        faces = pipeline.get_known_faces()
        logger.info(f"✓ Retrieved {len(faces)} faces")
        
        # Clean up
        pipeline.remove_known_face(face_id)
        
        # Clean up test database
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        
        return True
    except Exception as e:
        logger.error(f"✗ Add known face test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_process_frame():
    """Test processing a single frame."""
    logger.info("\n" + "="*50)
    logger.info("Testing Process Frame")
    logger.info("="*50)
    
    try:
        config = Config()
        config.DB_PATH = Path('data/test_pipeline.db')
        
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame (detection only, no recognition)
        results = pipeline.process_frame(frame, recognize=False)
        
        logger.info(f"✓ Frame processed")
        logger.info(f"  - Frame count: {results['frame_count']}")
        logger.info(f"  - Processing time: {results['processing_time']:.3f}s")
        logger.info(f"  - FPS: {results['fps']:.1f}")
        
        # Process another frame
        results2 = pipeline.process_frame(frame, recognize=False)
        logger.info(f"✓ Second frame processed")
        
        # Clean up test database
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        
        return True
    except Exception as e:
        logger.error(f"✗ Process frame test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization functions."""
    logger.info("\n" + "="*50)
    logger.info("Testing Visualization")
    logger.info("="*50)
    
    try:
        config = Config()
        config.DB_PATH = Path('data/test_pipeline.db')
        
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        results = pipeline.process_frame(frame, recognize=False)
        
        # Visualize
        output_frame = pipeline.visualize_results(frame, results)
        
        logger.info(f"✓ Visualization created")
        logger.info(f"  - Output shape: {output_frame.shape}")
        
        # Clean up test database
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        
        return True
    except Exception as e:
        logger.error(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_processor():
    """Test video processor."""
    logger.info("\n" + "="*50)
    logger.info("Testing Video Processor")
    logger.info("="*50)
    
    try:
        config = Config()
        config.DB_PATH = Path('data/test_pipeline.db')
        
        pipeline = RecognitionPipeline(config=config)
        processor = VideoProcessor(pipeline=pipeline, config=config)
        
        logger.info(f"✓ Video processor created")
        
        # Test with a few frames from camera (if available)
        # This is optional and may fail if no camera is available
        try:
            logger.info("Testing camera capture (may fail if no camera)...")
            
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                logger.info("✓ Camera available")
                cap.release()
            else:
                logger.info("✗ No camera available (this is OK)")
        except Exception as e:
            logger.info(f"✗ Camera test skipped: {e}")
        
        # Clean up test database
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        
        return True
    except Exception as e:
        logger.error(f"✗ Video processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_stats():
    """Test pipeline statistics."""
    logger.info("\n" + "="*50)
    logger.info("Testing Pipeline Statistics")
    logger.info("="*50)
    
    try:
        config = Config()
        config.DB_PATH = Path('data/test_pipeline.db')
        
        pipeline = RecognitionPipeline(config=config)
        pipeline.initialize()
        
        # Process a few frames
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(5):
            pipeline.process_frame(frame, recognize=False)
        
        # Get stats
        stats = pipeline.get_stats()
        
        logger.info(f"✓ Statistics retrieved:")
        logger.info(f"  - Frame count: {stats['frame_count']}")
        logger.info(f"  - Avg processing time: {stats['avg_processing_time']:.3f}s")
        logger.info(f"  - Avg FPS: {stats['avg_fps']:.1f}")
        logger.info(f"  - Known faces: {stats['known_faces']}")
        logger.info(f"  - Active tracks: {stats['active_tracks']}")
        
        # Clean up test database
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("FACE ID - RECOGNITION PIPELINE TESTS")
    logger.info("="*70)
    
    results = {
        'Pipeline Initialization': test_pipeline_initialization(),
        'Add Known Face': test_add_known_face(),
        'Process Frame': test_process_frame(),
        'Visualization': test_visualization(),
        'Video Processor': test_video_processor(),
        'Pipeline Stats': test_pipeline_stats()
    }
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name:30s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Recognition pipeline is ready.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
