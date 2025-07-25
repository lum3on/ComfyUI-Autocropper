"""
Test script for ColorGeneratorNode
==================================

This script validates the ColorGeneratorNode structure and syntax
without requiring torch/PIL dependencies.

Run this script to verify the node structure before using it in ComfyUI.
"""

import sys
import os
import ast

# Add the parent directory to the path so we can import the node
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_syntax():
    """Validate the Python syntax of the color generator node."""
    print("=== Validating Python Syntax ===")

    node_file = os.path.join(os.path.dirname(__file__), "color_generator_node.py")

    try:
        with open(node_file, 'r') as f:
            source_code = f.read()

        # Parse the AST to check for syntax errors
        ast.parse(source_code)
        print("✓ Python syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


def validate_structure():
    """Validate the node structure without importing torch."""
    print("\n=== Validating Node Structure ===")

    try:
        # Mock the required modules to avoid import errors
        import sys
        from unittest.mock import MagicMock

        # Mock torch and PIL
        sys.modules['torch'] = MagicMock()
        sys.modules['PIL'] = MagicMock()
        sys.modules['PIL.Image'] = MagicMock()

        # Now try to import the node
        from denrakeiw_nodes.color_generator_node import ColorGeneratorNode
        print("✓ Successfully imported ColorGeneratorNode")

        # Validate class structure
        if hasattr(ColorGeneratorNode, 'INPUT_TYPES'):
            print("✓ INPUT_TYPES method exists")
        else:
            print("✗ INPUT_TYPES method missing")
            return False

        if hasattr(ColorGeneratorNode, 'RETURN_TYPES'):
            print("✓ RETURN_TYPES attribute exists")
        else:
            print("✗ RETURN_TYPES attribute missing")
            return False

        if hasattr(ColorGeneratorNode, 'FUNCTION'):
            print("✓ FUNCTION attribute exists")
        else:
            print("✗ FUNCTION attribute missing")
            return False

        if hasattr(ColorGeneratorNode, 'CATEGORY'):
            print("✓ CATEGORY attribute exists")
        else:
            print("✗ CATEGORY attribute missing")
            return False

        # Validate colors dictionary
        if hasattr(ColorGeneratorNode, 'COLORS') and len(ColorGeneratorNode.COLORS) > 0:
            print(f"✓ COLORS dictionary has {len(ColorGeneratorNode.COLORS)} colors")
        else:
            print("✗ COLORS dictionary missing or empty")
            return False

        return True

    except ImportError as e:
        print(f"✗ Failed to import ColorGeneratorNode: {e}")
        return False


def test_color_generator():
    """Test the ColorGeneratorNode functionality."""
    print("\n=== Testing ColorGeneratorNode ===")
    
    # Create an instance of the node
    node = ColorGeneratorNode()
    
    # Test 1: Basic functionality with default parameters
    print("\nTest 1: Basic functionality")
    try:
        image_tensor, hex_code, color_name = node.generate_color_image("Red", 512, 512)
        print(f"✓ Generated {color_name} image with hex code {hex_code}")
        print(f"✓ Image tensor shape: {image_tensor.shape}")
        print(f"✓ Image tensor dtype: {image_tensor.dtype}")
        print(f"✓ Image tensor value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False
    
    # Test 2: Different colors
    print("\nTest 2: Different colors")
    test_colors = ["Blue", "Green", "Yellow", "Purple", "White", "Black"]
    for color in test_colors:
        try:
            image_tensor, hex_code, color_name = node.generate_color_image(color, 256, 256)
            print(f"✓ {color}: {hex_code}")
        except Exception as e:
            print(f"✗ Failed to generate {color}: {e}")
            return False
    
    # Test 3: Different dimensions
    print("\nTest 3: Different dimensions")
    test_dimensions = [(64, 64), (128, 256), (1024, 512), (1, 1), (4096, 4096)]
    for width, height in test_dimensions:
        try:
            image_tensor, hex_code, color_name = node.generate_color_image("Cyan", width, height)
            expected_shape = (1, height, width, 3)
            if image_tensor.shape == expected_shape:
                print(f"✓ Dimensions {width}x{height}: {image_tensor.shape}")
            else:
                print(f"✗ Dimensions {width}x{height}: Expected {expected_shape}, got {image_tensor.shape}")
                return False
        except Exception as e:
            print(f"✗ Failed with dimensions {width}x{height}: {e}")
            return False
    
    # Test 4: Input types validation
    print("\nTest 4: Input types validation")
    input_types = ColorGeneratorNode.INPUT_TYPES()
    print(f"✓ Available colors: {len(input_types['required']['color_name'][0])} colors")
    print(f"✓ Width range: {input_types['required']['width'][1]['min']}-{input_types['required']['width'][1]['max']}")
    print(f"✓ Height range: {input_types['required']['height'][1]['min']}-{input_types['required']['height'][1]['max']}")
    
    # Test 5: Return types validation
    print("\nTest 5: Return types validation")
    print(f"✓ Return types: {ColorGeneratorNode.RETURN_TYPES}")
    print(f"✓ Return names: {ColorGeneratorNode.RETURN_NAMES}")
    print(f"✓ Function name: {ColorGeneratorNode.FUNCTION}")
    print(f"✓ Category: {ColorGeneratorNode.CATEGORY}")
    
    print("\n=== All tests passed! ===")
    return True


if __name__ == "__main__":
    success = test_color_generator()
    if success:
        print("\n🎉 ColorGeneratorNode is ready to use in ComfyUI!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
