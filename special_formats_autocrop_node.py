import torch
from PIL import Image
import numpy as np
import base64
import io
import os
import cv2
import re
from dotenv import load_dotenv
from ultralytics import YOLO
import requests

# Load environment variables from .env file, overriding any existing ones
load_dotenv(override=True)

class SpecialFormatsAutoCropperNode:
    """
    Special Formats AutoCropper Node for advertising banner formats.
    Supports exact pixel dimensions for standard advertising placements.
    """
    
    # Define advertising format specifications with exact pixel dimensions
    ADVERTISING_FORMATS = {
        "Desktop Banner": {"width": 468, "height": 60, "aspect_ratio": 7.8, "use_case": "Standard web banner advertising"},
        "Desktop Leaderboard": {"width": 728, "height": 90, "aspect_ratio": 8.09, "use_case": "Top-of-page banner advertising"},
        "Desktop Skyscraper": {"width": 120, "height": 600, "aspect_ratio": 0.2, "use_case": "Sidebar vertical advertising"},
        "Desktop Large Skyscraper": {"width": 160, "height": 600, "aspect_ratio": 0.27, "use_case": "Wide sidebar vertical advertising"},
        "Mobile Leaderboard": {"width": 320, "height": 50, "aspect_ratio": 6.4, "use_case": "Mobile banner advertising"}
    }

    def __init__(self):
        self.model = self.load_model()

    def load_model(self, model_name="yolov8s.pt"):
        """Load YOLO model for object detection - same as original AutoCropperNode"""
        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}"
        model_dir = os.path.join("models", "ultralytics")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading model {model_name}...")
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as file:
                file.write(response.content)
            print(f"Model {model_name} downloaded successfully.")
        
        return YOLO(model_path)



    @classmethod
    def get_available_models(cls):
        """Get available models from Ollama and filter for vision-capable models"""
        try:
            import ollama
            from ollama import Client
            client = Client(host='http://localhost:11434')
            models_response = client.list()
            vision_models = []

            # Handle the actual Ollama response format
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                models_list = []

            # Known good vision models (prioritize these)
            known_vision_models = [
                'qwen2.5vl:7', 'qwen2.5vl:latest', 'qwen2-vl:latest',
                'llava:latest', 'llava:7b', 'llava:13b', 'llava:34b',
                'llama3.2-vision:latest', 'llama3.2-vision:11b', 'llama3.2-vision:90b',
                'minicpm-v:latest', 'moondream:latest'
            ]

            for model in models_list:
                # Extract model name from the model object
                if hasattr(model, 'model'):
                    model_name = model.model
                elif isinstance(model, dict) and 'model' in model:
                    model_name = model['model']
                else:
                    continue

                # Skip GGUF models and other incompatible formats
                if any(skip_pattern in model_name.lower() for skip_pattern in ['gguf/', 'onnx/', 'tensorrt/']):
                    continue

                # Prioritize known good vision models
                if model_name in known_vision_models:
                    vision_models.append(model_name)
                    continue

                # Check for vision keywords in model name
                vision_keywords = ['llava', 'vision', 'minicpm', 'moondream', 'janus', 'qwen', 'vl']
                if any(vision_keyword in model_name.lower() for vision_keyword in vision_keywords):
                    vision_models.append(model_name)

            # Remove duplicates while preserving order
            vision_models = list(dict.fromkeys(vision_models))

            if not vision_models:
                # Fallback to common vision models if none detected
                vision_models = ["qwen2.5vl:7", "qwen2.5vl:latest", "llava:latest", "llama3.2-vision:latest"]

            # Sort models to prioritize known working ones
            priority_order = ["qwen2.5vl:7", "qwen2.5vl:latest", "qwen2-vl:latest", "llama3.2-vision:latest", "llava:7b", "llava:13b", "llava:latest"]
            sorted_models = []

            # Add priority models first (if they exist)
            for priority_model in priority_order:
                if priority_model in vision_models:
                    sorted_models.append(priority_model)
                    vision_models.remove(priority_model)

            # Add remaining models
            sorted_models.extend(vision_models)

            return sorted_models
        except Exception as e:
            print(f"Could not fetch Ollama models: {e}")
            return ["qwen2.5vl:7", "qwen2.5vl:latest", "llava:latest", "llama3.2-vision:latest"]

    @classmethod
    def INPUT_TYPES(s):
        """Define input types for ComfyUI interface"""
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "This advertising-focused node uses Qwen vision model to analyze marketing images and intelligently crop them for standard advertising banner formats. It automatically selects the best format based on content analysis: products, logos, text, and brand elements to create effective advertising crops with exact pixel dimensions required by advertising platforms."
                }),
                "model": (s.get_available_models(),),
            },
            "optional": {
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
                "enable_reasoning": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cropped_image", "format_info", "reasoning", "raw_analysis_response")

    FUNCTION = "crop_for_advertising"

    CATEGORY = "Image"

    def analyze_advertising_content_with_ollama(self, img_base64, model, ollama_host="http://localhost:11434", temperature=0.1):
        """Use Ollama to analyze advertising content and automatically determine the best advertising format."""
        print(f"🎯 [ADVERTISING] Starting Ollama analysis with model: {model}")
        print(f"🎯 [ADVERTISING] Ollama host: {ollama_host}")
        print(f"🎯 [ADVERTISING] Temperature: {temperature}")

        try:
            import ollama
            from ollama import Client
            client = Client(host=ollama_host)
            print(f"🎯 [ADVERTISING] Ollama client created successfully")

            # Create format descriptions for the prompt
            format_descriptions = []
            for format_name, specs in self.ADVERTISING_FORMATS.items():
                format_descriptions.append(f"• {format_name}: {specs['width']}x{specs['height']} pixels (ratio {specs['aspect_ratio']:.2f}:1) - {specs['use_case']}")

            formats_text = "\n".join(format_descriptions)

            advertising_analysis_prompt = f"""You are an expert advertising format selector. Analyze this image and choose the BEST advertising format from the 5 available options. Be creative and consider the image composition carefully.

🎯 YOUR MISSION: Select the optimal advertising format that maximizes visual impact for this specific image.

📐 AVAILABLE FORMATS:
{formats_text}

🎨 SELECTION RULES (choose ONE format from the two options for each scene):

📸 PARKED SHOTS (static displays, showroom images, clean presentations):
   → Desktop Leaderboard (728×90) - for wide horizontal compositions
   → Desktop Skyscraper (120×600) - for vertical or portrait-oriented content

🏠 COCKPIT/INTERIOR (close-ups, interior details, confined spaces):
   → Desktop Large Skyscraper (160×600) - for detailed vertical content
   → Desktop Banner (468×60) - for horizontal dashboard/interior elements

🔍 DETAIL SHOTS (logos, badges, text, product features):
   → Desktop Banner (468×60) - for horizontal text/logo elements
   → Desktop Large Skyscraper (160×600) - for vertical detailed features

⚡ DYNAMIC SCENES (action, motion, wide compositions):
   → Desktop Leaderboard (728×90) - for wide action scenes
   → Mobile Leaderboard (320×50) - for mobile-optimized dynamic content

🧠 DECISION PROCESS:
1. Look at the image composition (horizontal vs vertical elements)
2. Identify the main subject and scene type
3. Consider which format would showcase the content best
4. Choose the format that maximizes visual impact

⚠️ IMPORTANT: Vary your selections! Don't always choose the same format. Consider the unique aspects of THIS specific image.

RESPOND IN THIS EXACT FORMAT:
SCENE_TYPE: [parked_shots, cockpit_interior, detail_shots, or dynamic_scenes]
SELECTED_FORMAT: [Desktop Banner, Desktop Leaderboard, Desktop Skyscraper, Desktop Large Skyscraper, or Mobile Leaderboard]
CROP_FOCUS: [describe the main subject/area for cropping]
KEY_ELEMENTS: [list important elements to preserve]
REASONING: [explain why this specific format works best for this image]"""

            print(f"🎯 [ADVERTISING] Prompt prepared, length: {len(advertising_analysis_prompt)} characters")

            # Try different image formats for Ollama compatibility
            response = None
            method_used = None

            try:
                print(f"🎯 [ADVERTISING] Method 1: Trying with base64 string...")
                response = client.chat(
                    model=model,
                    messages=[
                        {
                            'role': 'user',
                            'content': advertising_analysis_prompt,
                            'images': [img_base64]
                        }
                    ],
                    options={
                        'temperature': max(temperature, 0.7),  # Ensure minimum temperature for variety
                        'num_predict': 500,
                        'top_p': 0.9,  # Add top_p for more diverse responses
                        'repeat_penalty': 1.1,  # Reduce repetition
                    }
                )
                method_used = "base64_string"
                print(f"✅ [ADVERTISING] Method 1 successful!")

            except Exception as e1:
                print(f"❌ [ADVERTISING] Method 1 failed: {e1}")
                try:
                    print(f"🎯 [ADVERTISING] Method 2: Trying with decoded bytes...")
                    image_data = base64.b64decode(img_base64)
                    response = client.chat(
                        model=model,
                        messages=[
                            {
                                'role': 'user',
                                'content': advertising_analysis_prompt,
                                'images': [image_data]
                            }
                        ],
                        options={
                            'temperature': max(temperature, 0.7),  # Ensure minimum temperature for variety
                            'num_predict': 500,
                            'top_p': 0.9,  # Add top_p for more diverse responses
                            'repeat_penalty': 1.1,  # Reduce repetition
                        }
                    )
                    method_used = "decoded_bytes"
                    print(f"✅ [ADVERTISING] Method 2 successful!")

                except Exception as e2:
                    print(f"❌ [ADVERTISING] Method 2 failed: {e2}")
                    print(f"🎯 [ADVERTISING] Method 3: Trying text-only fallback...")
                    fallback_prompt = f"{advertising_analysis_prompt}\n\nNote: Image analysis not available, using text-only mode. Please provide a general advertising format recommendation."
                    response = client.chat(
                        model=model,
                        messages=[
                            {
                                'role': 'user',
                                'content': fallback_prompt
                            }
                        ],
                        options={
                            'temperature': max(temperature, 0.7),  # Ensure minimum temperature for variety
                            'num_predict': 500,
                            'top_p': 0.9,  # Add top_p for more diverse responses
                            'repeat_penalty': 1.1,  # Reduce repetition
                        }
                    )
                    method_used = "text_only"
                    print(f"⚠️ [ADVERTISING] Using text-only mode for model {model}")

            if response:
                content = response['message']['content']
                print(f"🎯 [ADVERTISING] Response received using method: {method_used}")
                print(f"🎯 [ADVERTISING] Response length: {len(content)} characters")
                print(f"🎯 [ADVERTISING] Response content:\n{content}")
                return content
            else:
                print(f"❌ [ADVERTISING] No response received from any method")
                return "No response received from Ollama"

        except Exception as e:
            print(f"❌ [ADVERTISING] Ollama analysis failed with exception: {e}")
            import traceback
            print(f"❌ [ADVERTISING] Full traceback:\n{traceback.format_exc()}")
            return f"Ollama analysis failed: {e}"

    def parse_advertising_analysis(self, ollama_response):
        """Parse the Ollama response to extract content type, selected format, crop focus, and reasoning."""
        print(f"🎯 [ADVERTISING] ===== PARSING OLLAMA RESPONSE =====")
        print(f"🎯 [ADVERTISING] Response length: {len(ollama_response) if ollama_response else 0} characters")
        print(f"🎯 [ADVERTISING] Response to parse: {repr(ollama_response)}")

        try:
            if not ollama_response or ollama_response.strip() == "":
                print(f"❌ [ADVERTISING] Empty response detected")
                # Use diverse fallback instead of always Desktop Leaderboard
                fallback_scene = "detail_shots"  # Changed from default "parked_shots"
                fallback_format = self._get_rule_based_fallback_format(fallback_scene, prefer_alternative=True)
                print(f"🔄 [ADVERTISING] Empty response fallback: {fallback_scene} → {fallback_format}")
                return fallback_scene, fallback_format, "center of image", "main visual elements", "Model returned empty response - using diverse fallback format"

            lines = ollama_response.strip().split('\n')
            print(f"🎯 [ADVERTISING] Split into {len(lines)} lines")
            print(f"🎯 [ADVERTISING] Lines: {[line.strip() for line in lines]}")

            scene_type = None
            selected_format = None
            crop_focus = None
            key_elements = None
            reasoning = None

            # Parse each line with detailed logging
            for i, line in enumerate(lines):
                line = line.strip()
                print(f"🎯 [ADVERTISING] Processing line {i+1}: {repr(line)}")

                if 'SCENE_TYPE:' in line:
                    scene_type = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"✅ [ADVERTISING] Found SCENE_TYPE: {repr(scene_type)}")
                elif 'SELECTED_FORMAT:' in line:
                    selected_format = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"✅ [ADVERTISING] Found SELECTED_FORMAT: {repr(selected_format)}")
                elif 'CROP_FOCUS:' in line:
                    crop_focus = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"✅ [ADVERTISING] Found CROP_FOCUS: {repr(crop_focus)}")
                elif 'KEY_ELEMENTS:' in line:
                    key_elements = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"✅ [ADVERTISING] Found KEY_ELEMENTS: {repr(key_elements)}")
                elif 'REASONING:' in line:
                    reasoning = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"✅ [ADVERTISING] Found REASONING: {repr(reasoning)}")
                else:
                    print(f"⚪ [ADVERTISING] Skipped line {i+1}: {repr(line)}")

            # Log what was extracted
            print(f"🎯 [ADVERTISING] ===== EXTRACTION RESULTS =====")
            print(f"🎯 [ADVERTISING] scene_type: {repr(scene_type)}")
            print(f"🎯 [ADVERTISING] selected_format: {repr(selected_format)}")
            print(f"🎯 [ADVERTISING] crop_focus: {repr(crop_focus)}")
            print(f"🎯 [ADVERTISING] key_elements: {repr(key_elements)}")
            print(f"🎯 [ADVERTISING] reasoning: {repr(reasoning)}")

            # Validate scene type with intelligent fallback
            valid_scene_types = ["parked_shots", "cockpit_interior", "detail_shots", "dynamic_scenes"]
            original_scene_type = scene_type
            if scene_type not in valid_scene_types:
                # Use diverse fallback scenes instead of always "parked_shots"
                import random
                fallback_scenes = ["detail_shots", "dynamic_scenes", "cockpit_interior", "parked_shots"]
                scene_type = random.choice(fallback_scenes)
                print(f"⚠️ [ADVERTISING] Invalid scene type '{original_scene_type}', using diverse fallback: {scene_type}")
            else:
                print(f"✅ [ADVERTISING] Valid scene type: {scene_type}")

            # Validate selected format with intelligent fallback
            valid_formats = list(self.ADVERTISING_FORMATS.keys())
            original_format = selected_format
            if selected_format not in valid_formats:
                # Rule-based fallback based on scene type
                selected_format = self._get_rule_based_fallback_format(scene_type)
                print(f"⚠️ [ADVERTISING] Invalid format '{original_format}', using rule-based fallback: {selected_format}")
            else:
                print(f"✅ [ADVERTISING] Valid format found: {selected_format}")
                # Validate that the selected format is allowed for this scene type
                if not self._is_format_valid_for_scene(scene_type, selected_format):
                    print(f"⚠️ [ADVERTISING] Format {selected_format} not optimal for scene {scene_type}")
                    # Try alternative format for this scene before falling back
                    selected_format = self._get_rule_based_fallback_format(scene_type, prefer_alternative=True)
                    print(f"🔄 [ADVERTISING] Using alternative format for scene: {selected_format}")
                else:
                    print(f"✅ [ADVERTISING] Format {selected_format} is valid for scene {scene_type}")

            print(f"🎯 [ADVERTISING] Final parsed values:")
            print(f"🎯 [ADVERTISING] - scene_type: {scene_type}")
            print(f"🎯 [ADVERTISING] - selected_format: {selected_format}")
            print(f"🎯 [ADVERTISING] - crop_focus: {crop_focus}")
            print(f"🎯 [ADVERTISING] - key_elements: {key_elements}")
            print(f"🎯 [ADVERTISING] - reasoning: {reasoning}")

            return scene_type, selected_format, crop_focus, key_elements, reasoning

        except Exception as e:
            print(f"❌ [ADVERTISING] Exception during parsing: {e}")
            import traceback
            print(f"❌ [ADVERTISING] Full traceback:\n{traceback.format_exc()}")
            # Use diverse fallback instead of always "parked_shots"
            import random
            fallback_scenes = ["dynamic_scenes", "detail_shots", "cockpit_interior", "parked_shots"]
            fallback_scene = random.choice(fallback_scenes)
            fallback_format = self._get_rule_based_fallback_format(fallback_scene, prefer_alternative=True)
            print(f"🔄 [ADVERTISING] Exception fallback: {fallback_scene} → {fallback_format}")
            return fallback_scene, fallback_format, "center of image", "main visual elements", f"Failed to parse Ollama response: {e}"

    def get_advertising_focus_point(self, img, crop_focus_description, yolo_results, scene_type):
        """Determine optimal crop center point for advertising content based on GPT analysis and YOLO detection."""
        width, height = img.size
        center_x, center_y = width / 2, height / 2  # Default center

        # Use YOLO results to find relevant objects for advertising
        if yolo_results and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes.xywh
            classes = yolo_results[0].boxes.cls
            class_names = yolo_results[0].names

            # Look for advertising-relevant objects
            relevant_boxes = []
            for box, cls_idx in zip(boxes, classes):
                class_name = class_names[int(cls_idx)]
                # Expand object detection for advertising content
                if class_name in ['person', 'bottle', 'cup', 'laptop', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']:
                    relevant_boxes.append(box)

            if relevant_boxes:
                # Use the largest relevant object as the focus point
                largest_object = max(relevant_boxes, key=lambda b: b[2] * b[3])
                x, y, _, _ = largest_object.cpu().numpy()
                center_x, center_y = x, y
            else:
                # If no specific objects found, use the largest detected object
                if len(boxes) > 0:
                    largest_box = max(boxes, key=lambda b: b[2] * b[3])
                    x, y, _, _ = largest_box.cpu().numpy()
                    center_x, center_y = x, y

        return center_x, center_y

    def _get_rule_based_fallback_format(self, scene_type, prefer_alternative=False):
        """Get format based on cropping rules for each scene type with intelligent selection"""
        import random

        # Cropping rules mapping - now with both options for intelligent selection
        scene_format_options = {
            "parked_shots": ["Desktop Leaderboard", "Desktop Skyscraper"],      # 728×90 or 120×600
            "cockpit_interior": ["Desktop Large Skyscraper", "Desktop Banner"], # 160×600 or 468×60
            "detail_shots": ["Desktop Banner", "Desktop Large Skyscraper"],     # 468×60 or 160×600
            "dynamic_scenes": ["Desktop Leaderboard", "Mobile Leaderboard"]     # 728×90 or 320×50
        }

        # Get valid options for this scene type
        options = scene_format_options.get(scene_type, ["Desktop Leaderboard", "Desktop Banner", "Desktop Skyscraper"])

        if prefer_alternative and len(options) > 1:
            # Prefer the second option to add variety
            selected = options[1]
            print(f"🎯 [RULE-BASED] Scene type '{scene_type}' → Alternative format: {selected}")
        else:
            # Use intelligent selection: 70% first option, 30% second option for variety
            if len(options) > 1 and random.random() < 0.3:
                selected = options[1]
                print(f"🎯 [RULE-BASED] Scene type '{scene_type}' → Random alternative format: {selected}")
            else:
                selected = options[0]
                print(f"🎯 [RULE-BASED] Scene type '{scene_type}' → Primary format: {selected}")

        return selected

    def _is_format_valid_for_scene(self, scene_type, selected_format):
        """Check if the selected format is valid according to cropping rules"""
        # Define valid format combinations for each scene type
        valid_combinations = {
            "parked_shots": ["Desktop Leaderboard", "Desktop Skyscraper"],
            "cockpit_interior": ["Desktop Large Skyscraper", "Desktop Banner"],
            "detail_shots": ["Desktop Banner", "Desktop Large Skyscraper"],
            "dynamic_scenes": ["Desktop Leaderboard", "Mobile Leaderboard"]
        }

        valid_formats = valid_combinations.get(scene_type, [])
        is_valid = selected_format in valid_formats

        if not is_valid:
            print(f"⚠️ [VALIDATION] Format '{selected_format}' not in valid options {valid_formats} for scene '{scene_type}'")
        else:
            print(f"✅ [VALIDATION] Format '{selected_format}' is valid for scene '{scene_type}'")

        return is_valid

    def crop_for_advertising(self, image, prompt, model, ollama_host="http://localhost:11434", enable_reasoning=True, temperature=0.8):
        """Main crop function for advertising formats with automatic format selection and exact pixel dimension output."""
        # 1. Convert the input tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # 2. Encode the image to base64 for Ollama analysis
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 3. YOLO Object Detection
        yolo_results = self.model.predict(img)

        # 4. Intelligent Format Selection (Ollama + Image Analysis Fallback)
        print(f"🎯 [ADVERTISING] ===== STARTING INTELLIGENT FORMAT SELECTION =====")
        print(f"🎯 [ADVERTISING] enable_reasoning: {enable_reasoning}")

        ollama_analysis = ""
        scene_type = None
        selected_format = None
        crop_focus = "center of image"
        key_elements = "main visual elements"
        reasoning = ""

        if enable_reasoning:
            print(f"🎯 [ADVERTISING] Attempting Ollama analysis...")
            try:
                ollama_analysis = self.analyze_advertising_content_with_ollama(img_base64, model, ollama_host, temperature)
                if ollama_analysis and "SCENE_TYPE:" in ollama_analysis:
                    print(f"✅ [ADVERTISING] Ollama analysis successful, parsing response...")
                    scene_type, selected_format, crop_focus, key_elements, reasoning = self.parse_advertising_analysis(ollama_analysis)
                    print(f"🎯 [ADVERTISING] Ollama result: scene={scene_type}, format={selected_format}")
                else:
                    print(f"⚠️ [ADVERTISING] Ollama analysis failed or returned invalid response")
                    ollama_analysis = f"Ollama failed: {ollama_analysis}"
                    scene_type, selected_format = None, None
            except Exception as e:
                print(f"❌ [ADVERTISING] Ollama analysis exception: {e}")
                ollama_analysis = f"Ollama exception: {e}"
                scene_type, selected_format = None, None

        # If Ollama failed or disabled, use intelligent image analysis fallback
        if not scene_type or not selected_format:
            print(f"🎯 [ADVERTISING] Using intelligent image analysis fallback...")
            scene_type, selected_format, reasoning = self._intelligent_format_selection_fallback(img, yolo_results)
            print(f"🎯 [ADVERTISING] Intelligent fallback result: scene={scene_type}, format={selected_format}")

        print(f"🎯 [ADVERTISING] Final selection: scene={scene_type}, format={selected_format}")

        # 5. Get format specifications for the automatically selected format
        format_spec = self.ADVERTISING_FORMATS[selected_format]
        target_width = format_spec["width"]
        target_height = format_spec["height"]
        target_aspect_ratio = format_spec["aspect_ratio"]

        print(f"🎯 [ADVERTISING] Auto-selected format: {selected_format}")
        print(f"🎯 [ADVERTISING] Target dimensions: {target_width}x{target_height}")
        print(f"🎯 [ADVERTISING] Scene type: {scene_type}")

        # 6. Get the crop focus point for advertising content
        center_x, center_y = self.get_advertising_focus_point(img, crop_focus, yolo_results, scene_type)

        # 7. Perform exact pixel dimension crop
        width, height = img.size
        image_aspect_ratio = float(width) / height

        # Calculate crop dimensions to match target aspect ratio
        if image_aspect_ratio > target_aspect_ratio:
            # Image is wider than target, crop horizontally
            new_height = height
            new_width = int(round(height * target_aspect_ratio))
        else:
            # Image is taller than target, crop vertically
            new_width = width
            new_height = int(round(width / target_aspect_ratio))

        # Center the crop on the determined focus point
        left = center_x - new_width / 2
        top = center_y - new_height / 2

        # Ensure crop stays within image boundaries
        left = max(0, min(left, width - new_width))
        top = max(0, min(top, height - new_height))

        # Define the final crop box as integers
        left, top = int(round(left)), int(round(top))
        right = left + new_width
        bottom = top + new_height

        # Perform the crop using PIL
        cropped_img = img.crop((left, top, right, bottom))

        # 8. Resize to exact pixel dimensions (this is the key difference from original)
        final_img = cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # 9. Validate exact pixel dimensions
        final_width, final_height = final_img.size
        if final_width != target_width or final_height != target_height:
            raise ValueError(f"Failed to achieve exact dimensions. Expected {target_width}x{target_height}, got {final_width}x{final_height}")

        # 10. Convert the final PIL Image back to a tensor
        output_image = np.array(final_img).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image).unsqueeze(0)

        # 11. Prepare detailed outputs
        format_info = f"Auto-Selected Format: {selected_format}\nDimensions: {target_width}x{target_height} pixels\nAspect Ratio: {target_aspect_ratio:.2f}:1\nUse Case: {format_spec['use_case']}\nScene Type: {scene_type}"

        detailed_reasoning = f"Scene Analysis: {scene_type}\nSelected Format: {selected_format}\nCrop Focus: {crop_focus}\nKey Elements: {key_elements}\nReasoning: {reasoning}"

        return (output_image, format_info, detailed_reasoning, ollama_analysis)

    def test_parsing_with_mock_responses(self):
        """Test the parsing logic with mock Ollama responses to verify it works correctly"""
        print(f"🧪 [TEST] ===== TESTING PARSING LOGIC =====")

        # Test case 1: Perfect response
        mock_response_1 = """SCENE_TYPE: detail_shots
SELECTED_FORMAT: Desktop Banner
CROP_FOCUS: product logo in center
KEY_ELEMENTS: brand logo, product text
REASONING: Detail shot with prominent logo requires banner format for text readability"""

        print(f"🧪 [TEST] Testing perfect response...")
        scene, format_sel, focus, elements, reason = self.parse_advertising_analysis(mock_response_1)
        print(f"🧪 [TEST] Result: {scene}, {format_sel}, {focus}, {elements}")

        # Test case 2: Response with extra text
        mock_response_2 = """Looking at this image, I can see it's a dynamic driving scene.

SCENE_TYPE: dynamic_scenes
SELECTED_FORMAT: Mobile Leaderboard
CROP_FOCUS: car in motion
KEY_ELEMENTS: vehicle, road, motion blur
REASONING: Dynamic scene with motion requires mobile format for action emphasis

This format will work well for mobile advertising."""

        print(f"🧪 [TEST] Testing response with extra text...")
        scene, format_sel, focus, elements, reason = self.parse_advertising_analysis(mock_response_2)
        print(f"🧪 [TEST] Result: {scene}, {format_sel}, {focus}, {elements}")

        # Test case 3: Malformed response
        mock_response_3 = """This is a cockpit interior shot.
Format: Desktop Large Skyscraper
Focus: dashboard area"""

        print(f"🧪 [TEST] Testing malformed response...")
        scene, format_sel, focus, elements, reason = self.parse_advertising_analysis(mock_response_3)
        print(f"🧪 [TEST] Result: {scene}, {format_sel}, {focus}, {elements}")

        # Test case 4: Empty response
        print(f"🧪 [TEST] Testing empty response...")
        scene, format_sel, focus, elements, reason = self.parse_advertising_analysis("")
        print(f"🧪 [TEST] Result: {scene}, {format_sel}, {focus}, {elements}")

        print(f"🧪 [TEST] ===== TESTING COMPLETE =====")
        return True

    def _intelligent_format_selection_fallback(self, img, yolo_results):
        """Intelligent format selection based on image analysis when Ollama is not available"""
        print(f"🧠 [INTELLIGENT] Starting image-based format selection...")

        width, height = img.size
        aspect_ratio = width / height

        # Analyze image characteristics
        image_analysis = self._analyze_image_characteristics(img, yolo_results)

        print(f"🧠 [INTELLIGENT] Image analysis: {image_analysis}")

        # Determine scene type based on image characteristics
        scene_type = self._determine_scene_type_from_analysis(image_analysis, aspect_ratio)

        # Select format based on scene type and image characteristics
        selected_format = self._select_format_for_scene(scene_type, image_analysis, aspect_ratio)

        reasoning = f"Image analysis: {image_analysis['primary_characteristic']}. Scene: {scene_type}. Format: {selected_format}"

        print(f"🧠 [INTELLIGENT] Final decision: {scene_type} → {selected_format}")
        return scene_type, selected_format, reasoning

    def _analyze_image_characteristics(self, img, yolo_results):
        """Analyze image characteristics to determine content type"""
        width, height = img.size
        aspect_ratio = width / height

        # Convert to numpy for analysis
        img_array = np.array(img)

        # Basic image analysis
        analysis = {
            'aspect_ratio': aspect_ratio,
            'is_wide': aspect_ratio > 1.5,
            'is_tall': aspect_ratio < 0.7,
            'is_square': 0.8 <= aspect_ratio <= 1.2,
            'has_objects': len(yolo_results[0].boxes) > 0 if yolo_results and len(yolo_results) > 0 else False,
            'object_count': len(yolo_results[0].boxes) if yolo_results and len(yolo_results) > 0 else 0,
            'primary_characteristic': 'unknown'
        }

        # Determine primary characteristic
        if analysis['object_count'] == 0:
            analysis['primary_characteristic'] = 'minimal_content'
        elif analysis['object_count'] == 1:
            analysis['primary_characteristic'] = 'single_subject'
        elif analysis['object_count'] <= 3:
            analysis['primary_characteristic'] = 'focused_composition'
        else:
            analysis['primary_characteristic'] = 'complex_scene'

        # Additional analysis based on aspect ratio
        if analysis['is_wide']:
            analysis['layout_preference'] = 'horizontal'
        elif analysis['is_tall']:
            analysis['layout_preference'] = 'vertical'
        else:
            analysis['layout_preference'] = 'balanced'

        return analysis

    def _determine_scene_type_from_analysis(self, analysis, aspect_ratio):
        """Determine scene type based on image analysis"""
        import random

        # Use image characteristics to intelligently guess scene type
        if analysis['primary_characteristic'] == 'minimal_content':
            # Minimal content suggests detail shots or clean product displays
            scene_options = ['detail_shots', 'parked_shots']
            scene_type = random.choice(scene_options)

        elif analysis['primary_characteristic'] == 'single_subject':
            # Single subject could be detail shot or parked shot
            if analysis['layout_preference'] == 'vertical':
                scene_type = 'detail_shots'  # Vertical suggests close-up details
            else:
                scene_type = 'parked_shots'  # Horizontal suggests product display

        elif analysis['primary_characteristic'] == 'complex_scene':
            # Complex scenes suggest dynamic or interior views
            if analysis['is_wide']:
                scene_type = 'dynamic_scenes'  # Wide complex scenes suggest action
            else:
                scene_type = 'cockpit_interior'  # Complex but not wide suggests interior

        else:  # focused_composition
            # Balanced composition - rotate through all types for variety
            scene_options = ['detail_shots', 'dynamic_scenes', 'cockpit_interior', 'parked_shots']
            scene_type = random.choice(scene_options)

        print(f"🧠 [INTELLIGENT] Scene determination: {analysis['primary_characteristic']} → {scene_type}")
        return scene_type

    def _select_format_for_scene(self, scene_type, analysis, aspect_ratio):
        """Select format based on scene type and image characteristics"""
        import random

        # Get valid formats for this scene type
        scene_format_options = {
            "parked_shots": ["Desktop Leaderboard", "Desktop Skyscraper"],
            "cockpit_interior": ["Desktop Large Skyscraper", "Desktop Banner"],
            "detail_shots": ["Desktop Banner", "Desktop Large Skyscraper"],
            "dynamic_scenes": ["Desktop Leaderboard", "Mobile Leaderboard"]
        }

        options = scene_format_options.get(scene_type, ["Desktop Banner", "Desktop Leaderboard"])

        # Use image characteristics to choose between the two options
        if len(options) == 2:
            format1, format2 = options[0], options[1]

            # Decision logic based on image characteristics
            if analysis['layout_preference'] == 'horizontal':
                # Prefer wider formats for horizontal layouts
                wide_formats = ['Desktop Leaderboard', 'Desktop Banner']
                if format1 in wide_formats:
                    selected_format = format1
                elif format2 in wide_formats:
                    selected_format = format2
                else:
                    selected_format = random.choice(options)

            elif analysis['layout_preference'] == 'vertical':
                # Prefer taller formats for vertical layouts
                tall_formats = ['Desktop Skyscraper', 'Desktop Large Skyscraper']
                if format1 in tall_formats:
                    selected_format = format1
                elif format2 in tall_formats:
                    selected_format = format2
                else:
                    selected_format = random.choice(options)

            else:
                # Balanced layout - use weighted random selection for variety
                if random.random() < 0.6:
                    selected_format = format1
                else:
                    selected_format = format2
        else:
            selected_format = options[0]

        print(f"🧠 [INTELLIGENT] Format selection: {scene_type} + {analysis['layout_preference']} → {selected_format}")
        return selected_format
