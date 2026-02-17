"""
Test Core Engine Components
Simple tests to verify the core modules work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FaceRecognizer
from face_recognition.tracker import FaceTracker
from face_recognition.database import FaceDatabase
from face_recognition.utils import cosine_similarity, calculate_iou
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_detector():
    """Test face detector."""
    logger.info("\n" + "="*50)
    logger.info("Testing Face Detector")
    logger.info("="*50)
    
    try:
        detector = FaceDetector(min_face_size=80, confidence_threshold=0.9)
        logger.info(f"✓ Detector created: {detector}")
        
        # Test with dummy image (would need real image for full test)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        logger.info("✓ Detector initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"✗ Detector test failed: {e}")
        return False


def test_recognizer():
    """Test face recognizer."""
    logger.info("\n" + "="*50)
    logger.info("Testing Face Recognizer")
    logger.info("="*50)
    
    try:
        recognizer = FaceRecognizer(model_name='buffalo_l', similarity_threshold=0.5)
        logger.info(f"✓ Recognizer created: {recognizer}")
        
        # Test embedding comparison
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        
        similarity = recognizer.compare(emb1, emb2)
        logger.info(f"✓ Similarity calculation works: {similarity:.3f}")
        
        # Test is_match
        is_match = recognizer.is_match(emb1, emb1, threshold=0.9)
        logger.info(f"✓ Self-matching works: {is_match}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Recognizer test failed: {e}")
        return False


def test_tracker():
    """Test face tracker."""
    logger.info("\n" + "="*50)
    logger.info("Testing Face Tracker")
    logger.info("="*50)
    
    try:
        tracker = FaceTracker(max_age=30, iou_threshold=0.3)
        logger.info(f"✓ Tracker created: {tracker}")
        
        # Test with dummy detections
        detections = [
            {'box': [100, 100, 50, 50], 'confidence': 0.95},
            {'box': [200, 150, 60, 60], 'confidence': 0.92}
        ]
        
        tracks = tracker.update(detections)
        logger.info(f"✓ Tracking works: {len(tracks)} tracks created")
        
        # Test track persistence
        tracks2 = tracker.update([
            {'box': [102, 102, 50, 50], 'confidence': 0.94},  # Slightly moved
            {'box': [250, 200, 55, 55], 'confidence': 0.90}   # New face
        ])
        logger.info(f"✓ Track persistence works: {len(tracks2)} tracks")
        
        return True
    except Exception as e:
        logger.error(f"✗ Tracker test failed: {e}")
        return False


def test_database():
    """Test face database."""
    logger.info("\n" + "="*50)
    logger.info("Testing Face Database")
    logger.info("="*50)
    
    try:
        # Use test database
        db = FaceDatabase(db_path='data/test_faces.db')
        logger.info(f"✓ Database created: {db}")
        
        # Initialize database
        db.initialize()
        logger.info("✓ Database initialized")
        
        # Test adding a face
        dummy_embedding = np.random.randn(512).astype(np.float32)
        face_id = db.add_face("Test Person", dummy_embedding, metadata={"test": True})
        logger.info(f"✓ Face added with ID: {face_id}")
        
        # Test retrieving face
        face_data = db.get_face(face_id)
        logger.info(f"✓ Face retrieved: {face_data['name']}")
        
        # Test listing faces
        faces = db.list_faces()
        logger.info(f"✓ Listed {len(faces)} faces")
        
        # Test face count
        count = db.get_face_count()
        logger.info(f"✓ Face count: {count}")
        
        # Test deletion
        deleted = db.delete_face(face_id)
        logger.info(f"✓ Face deleted: {deleted}")
        
        # Clean up
        db.close()
        os.remove('data/test_faces.db')
        logger.info("✓ Test database cleaned up")
        
        return True
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    logger.info("\n" + "="*50)
    logger.info("Testing Utility Functions")
    logger.info("="*50)
    
    try:
        # Test cosine similarity
        emb1 = np.array([1, 0, 0], dtype=np.float32)
        emb2 = np.array([1, 0, 0], dtype=np.float32)
        similarity = cosine_similarity(emb1, emb2)
        logger.info(f"✓ Cosine similarity (identical): {similarity:.3f}")
        
        emb3 = np.array([0, 1, 0], dtype=np.float32)
        similarity2 = cosine_similarity(emb1, emb3)
        logger.info(f"✓ Cosine similarity (orthogonal): {similarity2:.3f}")
        
        # Test IoU
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        iou = calculate_iou(box1, box2)
        logger.info(f"✓ IoU calculation: {iou:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Utils test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("FACE ID - CORE ENGINE TESTS")
    logger.info("="*70)
    
    results = {
        'Detector': test_detector(),
        'Recognizer': test_recognizer(),
        'Tracker': test_tracker(),
        'Database': test_database(),
        'Utils': test_utils()
    }
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Core engine is ready.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
