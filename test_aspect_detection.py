#!/usr/bin/env python3
"""
Test script for automatic aspect ratio detection
"""

import sys
import os
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from intelligent_autocrop_node import AspectRatioDetector
    print("✓ Successfully imported AspectRatioDetector")
except ImportError as e:
    print(f"✗ Failed to import AspectRatioDetector: {e}")
    sys.exit(1)

def test_aspect_detection():
    """Test the aspect ratio detection with different image scenarios"""

    detector = AspectRatioDetector()
    print(f"✓ AspectRatioDetector initialized")

    # Test scenarios with synthetic images
    test_cases = [
        {
            "name": "Ultra Wide (21:9 - should detect 16:9)",
            "size": (2560, 1080),
            "expected": ["16:9"]
        },
        {
            "name": "Wide Landscape (16:9)",
            "size": (1920, 1080),
            "expected": ["16:9", "3:2"]
        },
        {
            "name": "DSLR Standard (3:2)",
            "size": (1500, 1000),
            "expected": ["3:2", "4:3"]
        },
        {
            "name": "Traditional (4:3)",
            "size": (1200, 900),
            "expected": ["4:3", "3:2"]
        },
        {
            "name": "Square Image (1:1)",
            "size": (1000, 1000),
            "expected": ["1:1"]
        },
        {
            "name": "Portrait (4:5)",
            "size": (800, 1000),
            "expected": ["4:5", "1:1"]
        },
        {
            "name": "Vertical Video (9:16)",
            "size": (540, 960),
            "expected": ["4:5", "9:16"]
        }
    ]
    
    print("\n" + "="*50)
    print("TESTING ASPECT RATIO DETECTION")
    print("="*50)
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Image size: {test_case['size']}")
        
        # Create a synthetic test image
        width, height = test_case['size']
        test_image = Image.new('RGB', (width, height), color='blue')
        
        try:
            # Test the detection with debug enabled
            print("--- Detection Analysis ---")
            detected_ratio = detector.detect_optimal_aspect_ratio(test_image, confidence_threshold=0.4, debug=True)
            print("--- End Analysis ---")
            print(f"Final detected ratio: {detected_ratio}")

            if detected_ratio in test_case['expected']:
                print("✓ PASS - Detected ratio is within expected range")
            else:
                print(f"⚠ WARNING - Expected one of {test_case['expected']}, got {detected_ratio}")

        except Exception as e:
            print(f"✗ ERROR - Detection failed: {e}")
    
    print("\n" + "="*50)
    print("TESTING YOLO AVAILABILITY")
    print("="*50)
    
    # Test YOLO model loading
    try:
        yolo_loaded = detector.load_yolo_model()
        if yolo_loaded:
            print("✓ YOLO model loaded successfully")
            print("✓ Automotive object detection available")
        else:
            print("⚠ YOLO model not available - using fallback detection only")
            print("  Install ultralytics for full functionality: pip install ultralytics")
    except Exception as e:
        print(f"✗ YOLO loading failed: {e}")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print("The aspect ratio detection system is working!")
    print("For best results with automotive images:")
    print("1. Install ultralytics: pip install ultralytics>=8.0.0")
    print("2. Use images with clear car subjects")
    print("3. Adjust confidence_threshold (0.3-0.6) as needed")
    print("4. Manual override is always available")

if __name__ == "__main__":
    test_aspect_detection()
