"""
Color Generator Node for ComfyUI
================================

This node generates solid color images with customizable dimensions.
Supports 15 predefined colors and returns the image, hex code, and color name.

Features:
- 15 predefined colors (Red, Green, Blue, Yellow, Cyan, Magenta, etc.)
- Customizable width and height (1-4096 pixels)
- Returns image tensor, hex code, and color name
- Compatible with ComfyUI image processing pipeline

Author: denrakeiw
"""

import torch
import numpy as np
from PIL import Image


class ColorGeneratorNode:
    """
    A ComfyUI node that generates solid color images.
    
    This node creates images filled with a single color chosen from a predefined
    set of colors. The output includes the image tensor, hex color code, and 
    color name for further processing in ComfyUI workflows.
    """
    
    COLORS = {
        "Red": "#FF0000",
        "Green": "#00FF00", 
        "Blue": "#0000FF",
        "Yellow": "#FFFF00",
        "Cyan": "#00FFFF",
        "Magenta": "#FF00FF",
        "White": "#FFFFFF",
        "Black": "#000000",
        "Orange": "#FFA500",
        "Purple": "#800080",
        "Pink": "#FFC0CB",
        "Brown": "#A52A2A",
        "Gray": "#808080",
        "Navy": "#000080",
        "Lime": "#00FF00"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the node.
        
        Returns:
            dict: Input specification with required parameters
        """
        return {
            "required": {
                "color_name": (list(cls.COLORS.keys()), {"default": "Red"}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "hex_code", "color_name")
    FUNCTION = "generate_color_image"
    CATEGORY = "denrakeiw/image/generate"
    
    def generate_color_image(self, color_name, width, height):
        """
        Generate a solid color image with the specified parameters.
        
        Args:
            color_name (str): Name of the color from the predefined list
            width (int): Width of the generated image in pixels
            height (int): Height of the generated image in pixels
            
        Returns:
            tuple: (image_tensor, hex_code, color_name)
                - image_tensor: PyTorch tensor in ComfyUI format [1, H, W, 3]
                - hex_code: Hexadecimal color code (e.g., "#FF0000")
                - color_name: Name of the selected color
        """
        hex_code = self.COLORS[color_name]
        
        # Convert hex to RGB
        hex_code_clean = hex_code.lstrip('#')
        rgb = tuple(int(hex_code_clean[i:i+2], 16) for i in (0, 2, 4))
        
        # Create PIL image with the specified color
        pil_image = Image.new('RGB', (width, height), rgb)
        
        # Convert to tensor format expected by ComfyUI
        # ComfyUI expects images as float32 tensors with values in [0, 1]
        # and shape [batch, height, width, channels]
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]
        
        return (image_tensor, hex_code, color_name)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ColorGeneratorNode": ColorGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGeneratorNode": "Color Generator"
}
