import os
import subprocess
import sys

import importlib.util

def install_dependencies():
    # Check if required packages are installed
    required_packages = ["smartcrop", "ollama"]
    missing_packages = []

    for package in required_packages:
        if not importlib.util.find_spec(package):
            missing_packages.append(package)

    if not missing_packages:
        print("All required packages are already installed.")
        return

    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        print(f"Installing missing dependencies: {missing_packages}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")

install_dependencies()

"""
@author: Roo
@title: ComfyUI_AutoCropper
@nickname: AutoCropper
@description: This extension provides an automatic cropping node for ComfyUI, leveraging AI to suggest the best composition.
"""

# Import nodes with error handling
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import AutoCropperNode
try:
    from .autocrop_node import AutoCropperNode
    NODE_CLASS_MAPPINGS["AutoCropperNode"] = AutoCropperNode
    NODE_DISPLAY_NAME_MAPPINGS["AutoCropperNode"] = "Auto Crop Porsche"
except ImportError as e:
    print(f"Failed to import AutoCropperNode: {e}")

# Import IntelligentAutoCropperNode
try:
    from .intelligent_autocrop_node import IntelligentAutoCropperNode
    NODE_CLASS_MAPPINGS["IntelligentAutoCropperNode"] = IntelligentAutoCropperNode
    NODE_DISPLAY_NAME_MAPPINGS["IntelligentAutoCropperNode"] = "Intelligent Auto Cropper"
except ImportError as e:
    print(f"Failed to import IntelligentAutoCropperNode: {e}")

# Import OllamaAutoCropperNode
try:
    from .ollama_autocrop_node import OllamaAutoCropperNode
    NODE_CLASS_MAPPINGS["OllamaAutoCropperNode"] = OllamaAutoCropperNode
    NODE_DISPLAY_NAME_MAPPINGS["OllamaAutoCropperNode"] = "Ollama Auto Cropper"
    print("✓ OllamaAutoCropperNode loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import OllamaAutoCropperNode: {e}")
    print("Make sure 'ollama' package is installed: pip install ollama")

# Try to import AdvancedAutoCropperNode (may not exist)
try:
    from .advanced_autocrop_node import AdvancedAutoCropperNode
    NODE_CLASS_MAPPINGS["AdvancedAutoCropperNode"] = AdvancedAutoCropperNode
    NODE_DISPLAY_NAME_MAPPINGS["AdvancedAutoCropperNode"] = "Advanced Auto Cropper"
except ImportError:
    pass  # This file may not exist, so we ignore the error

# Import DynamicAutoCropperNode
try:
    from .dynamic_autocrop_node import DynamicAutoCropperNode
    NODE_CLASS_MAPPINGS["DynamicAutoCropperNode"] = DynamicAutoCropperNode
    NODE_DISPLAY_NAME_MAPPINGS["DynamicAutoCropperNode"] = "🎯 Dynamic Auto Cropper"
    print("✓ DynamicAutoCropperNode loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import DynamicAutoCropperNode: {e}")
    print("Make sure required packages are installed: ultralytics, opencv-python")

# Import TIFF Nodes
try:
    from .tiff_nodes import NODE_CLASS_MAPPINGS as TIFF_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TIFF_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(TIFF_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(TIFF_DISPLAY_MAPPINGS)
    print("✓ TIFF nodes loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import TIFF nodes: {e}")

# Import Denrakeiw Nodes
try:
    from .denrakeiw_nodes import NODE_CLASS_MAPPINGS as DENRAKEIW_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DENRAKEIW_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(DENRAKEIW_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(DENRAKEIW_DISPLAY_MAPPINGS)
    print("✓ Denrakeiw nodes loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import Denrakeiw nodes: {e}")

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']