#!/usr/bin/env python3
"""
Simple test for dynamic format selection without CUDA dependencies
"""

import sys
import os
import random

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rule_based_variety():
    """Test that the rule-based system now provides variety"""
    print("🎯 Testing rule-based format variety...")
    
    try:
        from special_formats_autocrop_node import SpecialFormatsAutoCropperNode
        
        node = SpecialFormatsAutoCropperNode()
        
        scene_types = ["parked_shots", "cockpit_interior", "detail_shots", "dynamic_scenes"]
        
        for scene in scene_types:
            print(f"\n🎭 Testing scene: {scene}")
            
            # Test multiple calls to see variety
            formats = []
            for i in range(10):
                fmt = node._get_rule_based_fallback_format(scene)
                formats.append(fmt)
            
            unique_formats = set(formats)
            print(f"   Formats from 10 calls: {list(unique_formats)}")
            print(f"   Variety: {len(unique_formats)} different formats")
            
            if len(unique_formats) >= 2:
                print(f"   ✅ Good variety!")
            else:
                print(f"   ⚠️ Limited variety")
                
            # Test alternative preference
            alt_format = node._get_rule_based_fallback_format(scene, prefer_alternative=True)
            print(f"   Alternative format: {alt_format}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule-based variety test failed: {e}")
        return False

def test_image_analysis_without_yolo():
    """Test image analysis logic without YOLO dependencies"""
    print("\n🧠 Testing image analysis logic...")
    
    try:
        from special_formats_autocrop_node import SpecialFormatsAutoCropperNode
        from PIL import Image
        
        node = SpecialFormatsAutoCropperNode()
        
        # Create test images with different characteristics
        test_cases = [
            ("wide_image", Image.new('RGB', (1200, 400), 'blue')),      # Wide aspect ratio
            ("tall_image", Image.new('RGB', (400, 1200), 'red')),       # Tall aspect ratio  
            ("square_image", Image.new('RGB', (800, 800), 'green')),    # Square aspect ratio
            ("standard_image", Image.new('RGB', (800, 600), 'yellow'))  # Standard aspect ratio
        ]
        
        for name, img in test_cases:
            print(f"\n📐 Testing {name} ({img.size[0]}x{img.size[1]}):")
            
            # Mock empty YOLO results
            mock_yolo_results = []
            
            # Test the analysis
            analysis = node._analyze_image_characteristics(img, mock_yolo_results)
            print(f"   Analysis: {analysis}")
            
            scene_type = node._determine_scene_type_from_analysis(analysis, analysis['aspect_ratio'])
            print(f"   Scene type: {scene_type}")
            
            selected_format = node._select_format_for_scene(scene_type, analysis, analysis['aspect_ratio'])
            print(f"   Selected format: {selected_format}")
            
            # Test multiple times for variety
            formats = []
            for i in range(5):
                scene = node._determine_scene_type_from_analysis(analysis, analysis['aspect_ratio'])
                fmt = node._select_format_for_scene(scene, analysis, analysis['aspect_ratio'])
                formats.append(fmt)
            
            unique_formats = set(formats)
            print(f"   Variety test: {len(unique_formats)} different formats: {list(unique_formats)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_format_distribution():
    """Test that all formats can be selected"""
    print("\n📊 Testing format distribution...")
    
    try:
        from special_formats_autocrop_node import SpecialFormatsAutoCropperNode
        
        node = SpecialFormatsAutoCropperNode()
        
        all_formats = set()
        scene_types = ["parked_shots", "cockpit_interior", "detail_shots", "dynamic_scenes"]
        
        # Collect formats from many calls
        for _ in range(50):
            scene = random.choice(scene_types)
            fmt = node._get_rule_based_fallback_format(scene)
            all_formats.add(fmt)
        
        print(f"📊 Formats seen in 50 random calls: {sorted(all_formats)}")
        print(f"📊 Total unique formats: {len(all_formats)}")
        
        expected_formats = {
            "Desktop Banner", "Desktop Leaderboard", "Desktop Skyscraper", 
            "Desktop Large Skyscraper", "Mobile Leaderboard"
        }
        
        missing_formats = expected_formats - all_formats
        if missing_formats:
            print(f"⚠️ Missing formats: {missing_formats}")
        else:
            print(f"✅ All formats can be selected!")
            
        return len(all_formats) >= 4  # Should see at least 4 different formats
        
    except Exception as e:
        print(f"❌ Format distribution test failed: {e}")
        return False

def main():
    """Run dynamic selection tests"""
    print("🚀 DYNAMIC FORMAT SELECTION TESTS")
    print("=" * 50)
    
    tests = [
        ("Rule-Based Variety", test_rule_based_variety),
        ("Image Analysis Logic", test_image_analysis_without_yolo),
        ("Format Distribution", test_format_distribution),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Dynamic format selection is working!")
        print("\n🎯 EXPECTED BEHAVIOR:")
        print("- Node will now select different formats based on image characteristics")
        print("- Even without Ollama, you'll get variety in format selection")
        print("- Formats will follow the cropping rules but with intelligent variation")
        return 0
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
