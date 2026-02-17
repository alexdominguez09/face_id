#!/usr/bin/env python3
"""
Quick verification script for face deduplication fix.
Tests the duplicate detection functionality.
"""

import sys
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from face_recognition import RecognitionPipeline, Config

def test_duplicate_detection():
    """Test that duplicate faces are detected and rejected."""

    print("=" * 70)
    print("FACE DEDUPLICATION VERIFICATION TEST")
    print("=" * 70)
    print()

    # Initialize pipeline
    config = Config()
    pipeline = RecognitionPipeline(config=config)
    pipeline.initialize()

    # Load a test image that we know exists
    test_image_path = "/home/alex/Downloads/people/train/images/person130.jpg"
    test_image = cv2.imread(test_image_path)

    if test_image is None:
        print(f"❌ Error: Could not load test image: {test_image_path}")
        return False

    print(f"Testing duplicate detection with: {test_image_path}")
    print()

    # Try to add duplicate face
    print("Test 1: Attempting to add duplicate face...")
    print("-" * 70)
    try:
        face_id = pipeline.add_known_face("DuplicateTest", test_image)
        print(f"❌ FAIL: Duplicate was added (ID: {face_id})")
        return False
    except ValueError as e:
        if "already exists" in str(e):
            print("✅ PASS: Duplicate correctly detected and rejected!")
            print()
            print(f"Error message: {e}")
            return True
        else:
            print(f"❌ FAIL: Wrong error - {e}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error - {e}")
        return False

def test_config_threshold():
    """Test that DUPLICATE_THRESHOLD is properly configured."""

    print()
    print("=" * 70)
    print("CONFIGURATION VERIFICATION")
    print("=" * 70)
    print()

    config = Config()

    if not hasattr(config, 'DUPLICATE_THRESHOLD'):
        print("❌ FAIL: DUPLICATE_THRESHOLD not found in config")
        return False

    print(f"✅ DUPLICATE_THRESHOLD found: {config.DUPLICATE_THRESHOLD}")
    print(f"✅ SIMILARITY_THRESHOLD: {config.SIMILARITY_THRESHOLD}")
    print()

    if config.DUPLICATE_THRESHOLD <= config.SIMILARITY_THRESHOLD:
        print("⚠️  WARNING: Duplicate threshold should be stricter than recognition threshold")
        print(f"   DUPLICATE_THRESHOLD ({config.DUPLICATE_THRESHOLD}) > SIMILARITY_THRESHOLD ({config.SIMILARITY_THRESHOLD})")
        return False
    else:
        print("✅ PASS: Duplicate threshold is stricter than recognition threshold")
        return True

def test_database_method():
    """Test that face_exists method is available in database."""

    print()
    print("=" * 70)
    print("DATABASE METHOD VERIFICATION")
    print("=" * 70)
    print()

    from face_recognition.database import FaceDatabase

    db = FaceDatabase()
    db.initialize()

    if not hasattr(db, 'face_exists'):
        print("❌ FAIL: face_exists method not found in FaceDatabase")
        return False

    print("✅ PASS: face_exists method available in FaceDatabase")
    print()

    # Test with a fake embedding
    import numpy as np
    fake_embedding = np.random.rand(512).astype(np.float32)

    result = db.face_exists(fake_embedding, threshold=0.85)

    if result is None:
        print("✅ PASS: Random embedding correctly identified as non-existent")
        return True
    else:
        print(f"❌ FAIL: Random embedding incorrectly matched face ID {result.get('id')}")
        return False

if __name__ == '__main__':
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║        FACE DEDUPLICATION FIX - VERIFICATION SCRIPT                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    results = []

    # Run all tests
    results.append(("Configuration", test_config_threshold()))
    results.append(("Database Method", test_database_method()))
    results.append(("Duplicate Detection", test_duplicate_detection()))

    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! Face deduplication is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Please review the implementation.")
        sys.exit(1)
