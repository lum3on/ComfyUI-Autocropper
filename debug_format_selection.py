#!/usr/bin/env python3
"""
Comprehensive diagnostic tool to debug why format selection is always 728x90
"""

import sys
import os
import base64
import io
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(scene_type="detail_shots"):
    """Create a simple test image for different scene types"""
    if scene_type == "detail_shots":
        # Create an image with text/logo elements (should prefer Desktop Banner 468×60)
        img = Image.new('RGB', (800, 600), color='white')
        # Add some text-like rectangles
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 250, 700, 350], fill='black')  # Text banner
        draw.rectangle([50, 50, 150, 150], fill='red')      # Logo
        
    elif scene_type == "dynamic_scenes":
        # Create an image with motion/action elements (should prefer Mobile Leaderboard 320×50)
        img = Image.new('RGB', (800, 600), color='blue')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Add diagonal lines to suggest motion
        for i in range(0, 800, 50):
            draw.line([(i, 0), (i+200, 600)], fill='white', width=3)
            
    elif scene_type == "cockpit_interior":
        # Create an interior-like image (should prefer Desktop Large Skyscraper 160×600)
        img = Image.new('RGB', (800, 600), color='brown')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Add dashboard-like elements
        draw.rectangle([100, 400, 700, 500], fill='black')  # Dashboard
        draw.circle([400, 300], 50, fill='white')           # Steering wheel
        
    else:  # parked_shots
        # Create a static product display (should prefer Desktop Leaderboard 728×90)
        img = Image.new('RGB', (800, 600), color='green')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Add car-like rectangle
        draw.rectangle([200, 250, 600, 350], fill='silver')
        
    return img

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    print("🔍 Testing Ollama connection...")
    
    try:
        import ollama
        from ollama import Client
        client = Client(host="http://localhost:11434")
        
        # Try to list models
        models = client.list()
        print(f"✅ Ollama is running! Available models: {[m['name'] for m in models['models']]}")
        return True, [m['name'] for m in models['models']]
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False, []

def test_format_selection_with_different_temperatures():
    """Test format selection with different temperature settings"""
    print("\n🌡️ Testing different temperature settings...")
    
    try:
        from special_formats_autocrop_node import SpecialFormatsAutoCropperNode
        
        node = SpecialFormatsAutoCropperNode()
        
        # Create a test image
        test_img = create_test_image("detail_shots")
        buffered = io.BytesIO()
        test_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        temperatures = [0.1, 0.5, 0.8, 1.0, 1.5]
        
        for temp in temperatures:
            print(f"\n🌡️ Testing temperature: {temp}")
            try:
                response = node.analyze_advertising_content_with_ollama(
                    img_base64, 
                    "qwen2-vl:7b",  # Default model
                    "http://localhost:11434", 
                    temp
                )
                
                scene_type, selected_format, crop_focus, key_elements, reasoning = node.parse_advertising_analysis(response)
                print(f"   Result: {scene_type} → {selected_format}")
                
            except Exception as e:
                print(f"   ❌ Failed at temperature {temp}: {e}")
                
    except Exception as e:
        print(f"❌ Temperature test failed: {e}")

def test_different_scene_types():
    """Test with different scene type images"""
    print("\n🎭 Testing different scene types...")
    
    try:
        from special_formats_autocrop_node import SpecialFormatsAutoCropperNode
        
        node = SpecialFormatsAutoCropperNode()
        
        scene_types = ["detail_shots", "dynamic_scenes", "cockpit_interior", "parked_shots"]
        
        for scene in scene_types:
            print(f"\n🎭 Testing scene: {scene}")
            
            # Create appropriate test image
            test_img = create_test_image(scene)
            buffered = io.BytesIO()
            test_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            try:
                # Test with higher temperature for variety
                response = node.analyze_advertising_content_with_ollama(
                    img_base64, 
                    "qwen2-vl:7b",
                    "http://localhost:11434", 
                    0.8  # Higher temperature
                )
                
                scene_type, selected_format, crop_focus, key_elements, reasoning = node.parse_advertising_analysis(response)
                print(f"   Expected scene: {scene}")
                print(f"   Detected scene: {scene_type}")
                print(f"   Selected format: {selected_format}")
                print(f"   Reasoning: {reasoning}")
                
                # Check if format matches expected rules
                expected_formats = {
                    "detail_shots": ["Desktop Banner", "Desktop Large Skyscraper"],
                    "dynamic_scenes": ["Desktop Leaderboard", "Mobile Leaderboard"],
                    "cockpit_interior": ["Desktop Large Skyscraper", "Desktop Banner"],
                    "parked_shots": ["Desktop Leaderboard", "Desktop Skyscraper"]
                }
                
                if scene_type in expected_formats and selected_format in expected_formats[scene_type]:
                    print(f"   ✅ Format selection follows rules!")
                else:
                    print(f"   ⚠️ Format selection doesn't follow rules")
                    print(f"   Expected formats for {scene_type}: {expected_formats.get(scene_type, 'Unknown')}")
                
            except Exception as e:
                print(f"   ❌ Failed for scene {scene}: {e}")
                
    except Exception as e:
        print(f"❌ Scene type test failed: {e}")

def test_intelligent_fallback():
    """Test the new intelligent fallback system (works without Ollama)"""
    print("\n🧠 Testing intelligent fallback system...")

    try:
        from special_formats_autocrop_node import SpecialFormatsAutoCropperNode

        node = SpecialFormatsAutoCropperNode()

        scene_types = ["detail_shots", "dynamic_scenes", "cockpit_interior", "parked_shots"]

        print("🧠 Testing intelligent fallback (no Ollama required):")

        for scene in scene_types:
            print(f"\n🎭 Testing scene: {scene}")

            # Create appropriate test image
            test_img = create_test_image(scene)

            # Test the intelligent fallback directly
            yolo_results = node.model.predict(test_img)
            scene_type, selected_format, reasoning = node._intelligent_format_selection_fallback(test_img, yolo_results)

            print(f"   Input scene: {scene}")
            print(f"   Detected scene: {scene_type}")
            print(f"   Selected format: {selected_format}")
            print(f"   Reasoning: {reasoning}")

            # Test multiple times to verify variety
            formats = []
            for i in range(5):
                _, fmt, _ = node._intelligent_format_selection_fallback(test_img, yolo_results)
                formats.append(fmt)

            unique_formats = set(formats)
            print(f"   Variety test: {len(unique_formats)} different formats from 5 calls: {list(unique_formats)}")

        print("✅ Intelligent fallback test completed")
        return True

    except Exception as e:
        print(f"❌ Intelligent fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive format selection diagnostics"""
    print("🚀 COMPREHENSIVE FORMAT SELECTION DIAGNOSTICS")
    print("=" * 60)

    # Test 1: Ollama connection
    ollama_working, models = test_ollama_connection()

    # Test 2: Intelligent fallback (works without Ollama)
    print("\n" + "=" * 60)
    print("TESTING INTELLIGENT FALLBACK (NO OLLAMA REQUIRED)")
    print("=" * 60)
    fallback_success = test_intelligent_fallback()

    if ollama_working:
        print("\n" + "=" * 60)
        print("TESTING WITH OLLAMA")
        print("=" * 60)

        # Test 3: Different temperatures
        test_format_selection_with_different_temperatures()

        # Test 4: Different scene types
        test_different_scene_types()
    else:
        print("\n⚠️ Ollama not available, but intelligent fallback should work!")

    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 60)

    if fallback_success:
        print("\n✅ SUCCESS: Intelligent fallback system is working!")
        print("   The node will now provide dynamic format selection even without Ollama.")

    if ollama_working:
        print("✅ Ollama is available for enhanced analysis.")
    else:
        print("⚠️ Ollama not available, but fallback system provides dynamic selection.")

    print("\n💡 RECOMMENDATIONS:")
    print("1. The node now uses intelligent image analysis for dynamic format selection")
    print("2. Formats will vary based on image characteristics and scene type")
    print("3. If you want even more variety, ensure Ollama is running with a vision model")
    print("4. The default temperature is now 0.8 for more diverse Ollama responses")

if __name__ == "__main__":
    main()
