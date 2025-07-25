"""
Denrakeiw Nodes - A ComfyUI Node Pack
=====================================

This package contains custom nodes for ComfyUI, including:
- ColorGeneratorNode: Generate solid color images with customizable dimensions

Author: denrakeiw
Version: 1.0.0
"""

from .color_generator_node import ColorGeneratorNode

# Export all node classes
NODE_CLASS_MAPPINGS = {
    "ColorGeneratorNode": ColorGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGeneratorNode": "Color Generator"
}

# Package metadata
__version__ = "1.0.0"
__author__ = "denrakeiw"
__description__ = "Custom ComfyUI nodes for image generation and manipulation"

# Make sure all nodes are properly exported
__all__ = [
    "ColorGeneratorNode",
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS"
]
