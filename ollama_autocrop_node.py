import torch
from PIL import Image
import numpy as np
import base64
import io
import os
import cv2
import re
from ultralytics import YOLO
import requests
import ollama
from ollama import Client

class OllamaAutoCropperNode:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self, model_name="yolov8s.pt"):
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
                vision_keywords = ['llava', 'vision', 'minicpm', 'moondream', 'janus', 'qwen']
                if any(vision_keyword in model_name.lower() for vision_keyword in vision_keywords):
                    vision_models.append(model_name)

            # Remove duplicates while preserving order
            vision_models = list(dict.fromkeys(vision_models))

            if not vision_models:
                # Fallback to common vision models if none detected
                vision_models = ["llava:latest", "llava:7b", "llava:13b", "llama3.2-vision:latest"]

            # Sort models to prioritize known working ones
            priority_order = ["qwen2.5vl:7", "qwen2.5vl:latest", "llama3.2-vision:latest", "llava:7b", "llava:13b", "llava:latest"]
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
            return ["llava:latest", "llava:7b", "llava:13b", "llama3.2-vision:latest", "qwen2.5vl:7", "qwen2.5vl:latest"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "This automotive-focused node uses Ollama to analyze car images with enhanced precision and intelligently crop them based on detailed scene analysis. It uses advanced detection for: Detail shots (badges, wheels, lights - 1:1/4:5), Cockpit/interior views (PREFER 4:5 for optimal dashboard framing, alternative 1:1), Parked shots (16:9/1:1), Driving distant shots (9:16/4:5), Dynamic driving scenes (16:9/9:16). The analysis prioritizes scale, perspective, and subject matter for accurate scene classification."
                }),
                "model": (s.get_available_models(),),
            },
            "optional": {
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
                "enable_reasoning": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("cropped_image", "reasoning", "raw_api_response")

    FUNCTION = "crop"

    CATEGORY = "Image"

    def analyze_scene_with_ollama(self, img_base64, model, ollama_host="http://localhost:11434", temperature=0.1):
        """Use Ollama to analyze the scene and determine the best aspect ratio based on automotive cropping rules."""
        print(f"🔍 [VERBOSE] Starting Ollama analysis with model: {model}")
        print(f"🔍 [VERBOSE] Ollama host: {ollama_host}")
        print(f"🔍 [VERBOSE] Temperature: {temperature}")
        print(f"🔍 [VERBOSE] Image data length: {len(img_base64)} characters")

        try:
            client = Client(host=ollama_host)
            print(f"🔍 [VERBOSE] Ollama client created successfully")

            scene_analysis_prompt = """CRITICAL: Analyze this automotive image very carefully. Look at the OVERALL SCENE CONTEXT first.

STEP 1 - IDENTIFY THE SCENE CONTEXT:
What type of environment/setting do you see?
- Are you looking FROM INSIDE a car (dashboard/steering wheel visible)?
- Do you see ROADS, STREETS, or DRIVING ENVIRONMENT?
- Is this a CLOSE-UP of a single car part (badge, wheel, etc.)?
- Is this a STATIONARY car display/parking scene?

DECISION TREE - ANSWER IN ORDER:

1️⃣ INTERIOR CHECK: Can you see dashboard, steering wheel, seats together?
   → IF YES: Continue to 1a and 1b

   1a) DYNAMIC_COCKPIT - Interior view WITH motion/action elements?
       - Dashboard view with motion blur or speed indicators?
       - Racing cockpit with dramatic angles or movement?
       - Interior shot during high-speed driving or action?
       - Cockpit view with visible speed/action context?
       → IF YES: DYNAMIC_COCKPIT

   1b) COCKPIT_INTERIOR - Static/composed interior view?
       - Clean dashboard view without motion blur?
       - Stationary interior photography?
       - Composed cockpit shot for display/review?
       → IF YES: COCKPIT_INTERIOR

2️⃣ DRIVING CONTEXT CHECK: Do you see roads, streets, or driving environment?
   - Car on a road/street with visible pavement?
   - Trees, buildings, or landscape around a moving/positioned car?
   - Environmental context suggesting the car is "out and about"?
   → IF YES: Continue to 2a and 2b

   2a) DYNAMIC_DRIVING - Is there obvious motion, action, or dramatic angles?
       - Motion blur visible?
       - Car cornering, racing, or at speed?
       - Dramatic low/high angles emphasizing action?
       - Close-up action shot with movement?
       → IF YES: DYNAMIC_DRIVING

   2b) DRIVING_DISTANT - Is this a more static/composed driving scene?
       - Car positioned normally on road/street?
       - Environmental context prominent (landscape, buildings)?
       - No obvious motion blur or dramatic action?
       - Car appears stationary or at normal speed?
       → IF YES: DRIVING_DISTANT

3️⃣ DETAIL CHECK: Is this a close-up of ONE specific car component?
   - Single badge, wheel, light, or mechanical part?
   - Component fills most of the frame?
   - No environmental context visible?
   → IF YES: DETAIL

4️⃣ STATIONARY CHECK: Is the car parked/displayed in a non-driving context?
   - Car in showroom, garage, or dealership?
   - Car in parking lot, driveway, or parking space?
   - Clean, controlled, or organized environment?
   - Car positioned for viewing/photography/display?
   - No road/street driving context visible?
   - Static presentation of the vehicle?
   → IF YES: PARKED

CRITICAL RULES:
🛣️ DRIVING SCENES - Two Types:

DYNAMIC_DRIVING (Action/Motion):
- Motion blur, speed lines, or movement visible
- Dramatic angles (low, high, tilted)
- Car cornering, drifting, racing
- Action-focused composition
- Emphasis on speed/movement

DRIVING_DISTANT (Static/Environmental):
- Car positioned normally on road/street
- No obvious motion blur or dramatic action
- Environmental context prominent
- Composed, stable shot
- Car appears stationary or normal speed
- Landscape/cityscape driving scenes

🏠 INTERIOR: Multiple interior elements together (dashboard + wheel + seats)

🔍 DETAIL: ONLY single components with NO environmental context

🚗 PARKED: Stationary display settings (showroom, garage, clean parking)

EXAMPLES:

🏁 DYNAMIC_DRIVING (Action/Motion):
- Car cornering with motion blur
- Racing scene with dramatic angle
- Car drifting or at high speed
- Action shot with movement visible
- Close-up of car in motion

🛣️ DRIVING_DISTANT (Static/Composed):
- Car positioned normally on road with palm trees
- Car on street with buildings in background
- Car driving normally without motion blur
- Environmental/landscape driving scene
- Car appears stationary or at normal speed

🔍 DETAIL: Close-up of Porsche badge only (no environment)
🏠 COCKPIT_INTERIOR: Dashboard view from inside
🚗 PARKED: Car in showroom, parking lot, driveway, or display setting

COCKPIT RATIO ANALYSIS (for COCKPIT_INTERIOR scenes):
Analyze these specific elements to choose between 4:5 and 1:1:

🎯 DASHBOARD LAYOUT:
- WIDE_DASHBOARD: Horizontal instrument clusters, wide center console → 4:5
- TALL_DASHBOARD: Vertical elements, tall center stack → 1:1
- BALANCED_DASHBOARD: Equal width/height elements → 4:5 (default preference)

🎯 STEERING WHEEL POSITION:
- CENTERED_WHEEL: Steering wheel centered in frame → 1:1
- PARTIAL_WHEEL: Steering wheel partially visible/off-center → 4:5
- NO_WHEEL: Dashboard only, no steering wheel → analyze dashboard layout

🎯 COMPOSITION ELEMENTS:
- SEATS_VISIBLE: Driver/passenger seats in frame → 4:5 (captures context)
- DOOR_PANELS: Side door panels/windows visible → 4:5 (wider context)
- INSTRUMENT_FOCUS: Close focus on instrument cluster only → 1:1
- FULL_CABIN: Wide cabin view with multiple elements → 4:5

RESPOND WITH COCKPIT ANALYSIS:
**DASHBOARD_TYPE:** [wide_dashboard, tall_dashboard, balanced_dashboard]
**WHEEL_POSITION:** [centered_wheel, partial_wheel, no_wheel]
**COMPOSITION:** [seats_visible, door_panels, instrument_focus, full_cabin]
**COCKPIT_RATIO_REASONING:** [detailed analysis of why 4:5 or 1:1 is better]

CROPPING RULES:
• Cockpit/interior: **DYNAMIC SELECTION** - 4:5 (default preference) or 1:1 (based on analysis)
• Detail shots: 1:1 or 4:5
• Parked shots: 16:9 or 1:1
• Driving distant: 4:5 or 16:9
• Dynamic driving: 16:9 or 9:16

RESPOND EXACTLY:
SCENE_TYPE: [cockpit_interior, detail, parked, driving_distant, dynamic_driving]
RECOMMENDED_RATIO: [16:9, 9:16, 1:1, 4:5]
CROP_FOCUS: [main subject description]
REASONING: [why this classification and ratio]"""

            print(f"🔍 [VERBOSE] Prompt prepared, length: {len(scene_analysis_prompt)} characters")

            # Try different image formats for Ollama compatibility
            response = None
            method_used = None

            try:
                print(f"🔍 [VERBOSE] Method 1: Trying with base64 string...")
                # Method 1: Try with base64 string (most common)
                response = client.chat(
                    model=model,
                    messages=[
                        {
                            'role': 'user',
                            'content': scene_analysis_prompt,
                            'images': [img_base64]
                        }
                    ],
                    options={
                        'temperature': temperature,
                        'num_predict': 400,
                    }
                )
                method_used = "base64_string"
                print(f"✅ [VERBOSE] Method 1 successful!")

            except Exception as e1:
                print(f"❌ [VERBOSE] Method 1 failed: {e1}")
                try:
                    print(f"🔍 [VERBOSE] Method 2: Trying with decoded bytes...")
                    # Method 2: Try with bytes
                    image_data = base64.b64decode(img_base64)
                    print(f"🔍 [VERBOSE] Decoded image data length: {len(image_data)} bytes")
                    response = client.chat(
                        model=model,
                        messages=[
                            {
                                'role': 'user',
                                'content': scene_analysis_prompt,
                                'images': [image_data]
                            }
                        ],
                        options={
                            'temperature': temperature,
                            'num_predict': 400,
                        }
                    )
                    method_used = "decoded_bytes"
                    print(f"✅ [VERBOSE] Method 2 successful!")

                except Exception as e2:
                    print(f"❌ [VERBOSE] Method 2 failed: {e2}")
                    print(f"🔍 [VERBOSE] Method 3: Trying text-only fallback...")
                    # Method 3: Try without images (text-only fallback)
                    fallback_prompt = f"{scene_analysis_prompt}\n\nNote: Image analysis not available, using text-only mode. Please provide a general automotive crop recommendation."
                    response = client.chat(
                        model=model,
                        messages=[
                            {
                                'role': 'user',
                                'content': fallback_prompt
                            }
                        ],
                        options={
                            'temperature': temperature,
                            'num_predict': 400,
                        }
                    )
                    method_used = "text_only"
                    print(f"⚠️ [VERBOSE] Using text-only mode for model {model}")
                    print(f"⚠️ [VERBOSE] Original errors - Method 1: {e1}, Method 2: {e2}")

            if response:
                content = response['message']['content']
                print(f"🔍 [VERBOSE] Response received using method: {method_used}")
                print(f"🔍 [VERBOSE] Response length: {len(content)} characters")
                print(f"🔍 [VERBOSE] Raw response: {repr(content)}")
                print(f"🔍 [VERBOSE] Response content:\n{content}")
                return content
            else:
                print(f"❌ [VERBOSE] No response received from any method")
                return "No response received from Ollama"

        except Exception as e:
            print(f"❌ [VERBOSE] Ollama analysis failed with exception: {e}")
            import traceback
            print(f"❌ [VERBOSE] Full traceback:\n{traceback.format_exc()}")
            return f"Ollama analysis failed: {e}"

    def analyze_cockpit_layout(self, ollama_response):
        """Extract detailed cockpit analysis from Ollama response for intelligent ratio selection."""
        cockpit_analysis = {
            'dashboard_type': 'balanced_dashboard',
            'wheel_position': 'partial_wheel',
            'composition': 'full_cabin',
            'cockpit_reasoning': ''
        }

        try:
            lines = ollama_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if 'DASHBOARD_TYPE:' in line:
                    cockpit_analysis['dashboard_type'] = line.split(':', 1)[1].replace('*', '').strip().lower()
                elif 'WHEEL_POSITION:' in line:
                    cockpit_analysis['wheel_position'] = line.split(':', 1)[1].replace('*', '').strip().lower()
                elif 'COMPOSITION:' in line:
                    cockpit_analysis['composition'] = line.split(':', 1)[1].replace('*', '').strip().lower()
                elif 'COCKPIT_RATIO_REASONING:' in line:
                    cockpit_analysis['cockpit_reasoning'] = line.split(':', 1)[1].replace('*', '').strip()
        except Exception as e:
            print(f"⚠️ [VERBOSE] Could not parse cockpit analysis: {e}")

        return cockpit_analysis

    def intelligent_cockpit_ratio_selection(self, cockpit_analysis, context_info):
        """Intelligently select between 4:5 and 1:1 for cockpit shots based on detailed analysis."""
        print(f"🧠 [COCKPIT] Analyzing layout for intelligent ratio selection...")
        print(f"🧠 [COCKPIT] Dashboard: {cockpit_analysis['dashboard_type']}")
        print(f"🧠 [COCKPIT] Wheel: {cockpit_analysis['wheel_position']}")
        print(f"🧠 [COCKPIT] Composition: {cockpit_analysis['composition']}")

        # Scoring system for ratio selection
        ratio_4_5_score = 60  # Default preference for 4:5
        ratio_1_1_score = 40

        # Dashboard layout analysis
        if cockpit_analysis['dashboard_type'] == 'wide_dashboard':
            ratio_4_5_score += 20
            print(f"🧠 [COCKPIT] Wide dashboard detected: +20 for 4:5")
        elif cockpit_analysis['dashboard_type'] == 'tall_dashboard':
            ratio_1_1_score += 25
            print(f"🧠 [COCKPIT] Tall dashboard detected: +25 for 1:1")
        else:  # balanced_dashboard
            ratio_4_5_score += 10
            print(f"🧠 [COCKPIT] Balanced dashboard: +10 for 4:5 (default preference)")

        # Steering wheel position analysis
        if cockpit_analysis['wheel_position'] == 'centered_wheel':
            ratio_1_1_score += 20
            print(f"🧠 [COCKPIT] Centered steering wheel: +20 for 1:1")
        elif cockpit_analysis['wheel_position'] == 'partial_wheel':
            ratio_4_5_score += 15
            print(f"🧠 [COCKPIT] Partial steering wheel: +15 for 4:5")
        # no_wheel doesn't add points, relies on dashboard analysis

        # Composition analysis
        if cockpit_analysis['composition'] in ['seats_visible', 'door_panels', 'full_cabin']:
            ratio_4_5_score += 15
            print(f"🧠 [COCKPIT] Wide composition elements: +15 for 4:5")
        elif cockpit_analysis['composition'] == 'instrument_focus':
            ratio_1_1_score += 20
            print(f"🧠 [COCKPIT] Instrument focus: +20 for 1:1")

        # Context from YOLO detection
        if context_info.get('car_coverage', 0) > 0.3:
            ratio_4_5_score += 5
            print(f"🧠 [COCKPIT] High car coverage: +5 for 4:5")

        # Final decision
        selected_ratio = '4:5' if ratio_4_5_score >= ratio_1_1_score else '1:1'
        confidence = max(ratio_4_5_score, ratio_1_1_score)

        print(f"🧠 [COCKPIT] Final scores - 4:5: {ratio_4_5_score}, 1:1: {ratio_1_1_score}")
        print(f"🧠 [COCKPIT] Selected ratio: {selected_ratio} (confidence: {confidence})")

        return selected_ratio, confidence

    def parse_ollama_analysis(self, ollama_response):
        """Parse the Ollama response to extract scene type, recommended ratio, and reasoning."""
        print(f"🔍 [VERBOSE] Parsing Ollama response...")
        print(f"🔍 [VERBOSE] Response to parse: {repr(ollama_response)}")
        print(f"🔍 [VERBOSE] Response length: {len(ollama_response) if ollama_response else 0}")

        try:
            if not ollama_response or ollama_response.strip() == "":
                print(f"❌ [VERBOSE] Empty or None response received - model may not support vision or prompt format")
                print(f"❌ [VERBOSE] Falling back to default automotive crop settings")
                return "unknown", "1:1", "center of image", "Model returned empty response - using default automotive settings (unknown scene, 1:1 ratio)"

            lines = ollama_response.strip().split('\n')
            print(f"🔍 [VERBOSE] Split into {len(lines)} lines:")
            for i, line in enumerate(lines):
                print(f"🔍 [VERBOSE] Line {i}: {repr(line)}")

            scene_type = None
            recommended_ratio = None
            crop_focus = None
            reasoning = None

            for line in lines:
                line = line.strip()
                # Handle multiple formats: "SCENE_TYPE:", "**Scene Type:**", "**SCENE_TYPE:**", "**SCENE_TYPE: value**"
                if ('SCENE_TYPE:' in line):
                    # Extract everything after the colon, removing asterisks
                    scene_type = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"🔍 [VERBOSE] Found SCENE_TYPE: {repr(scene_type)}")
                elif ('RECOMMENDED_RATIO:' in line):
                    # Extract everything after the colon, removing asterisks
                    recommended_ratio = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"🔍 [VERBOSE] Found RECOMMENDED_RATIO: {repr(recommended_ratio)}")
                elif ('CROP_FOCUS:' in line):
                    # Extract everything after the colon, removing asterisks
                    crop_focus = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"🔍 [VERBOSE] Found CROP_FOCUS: {repr(crop_focus)}")
                elif ('REASONING:' in line):
                    # Extract everything after the colon, removing asterisks
                    reasoning = line.split(':', 1)[1].replace('*', '').strip()
                    print(f"🔍 [VERBOSE] Found REASONING: {repr(reasoning)}")

            print(f"🔍 [VERBOSE] Parsed values:")
            print(f"🔍 [VERBOSE] - scene_type: {scene_type}")
            print(f"🔍 [VERBOSE] - recommended_ratio: {recommended_ratio}")
            print(f"🔍 [VERBOSE] - crop_focus: {crop_focus}")
            print(f"🔍 [VERBOSE] - reasoning: {reasoning}")

            # Clean up and validate the recommended ratio
            if recommended_ratio:
                recommended_ratio = recommended_ratio.strip()
                print(f"🔍 [VERBOSE] After cleanup, ratio is: {repr(recommended_ratio)}")

            # Scene-specific ratio validation
            scene_ratio_rules = {
                'cockpit_interior': ['4:5', '1:1'],  # Dynamic selection for cockpit views
                'detail': ['1:1', '4:5'],
                'parked': ['16:9', '1:1'],
                'driving_distant': ['4:5', '16:9'],
                'dynamic_driving': ['16:9', '9:16']
            }

            valid_ratios = ["16:9", "9:16", "1:1", "4:5"]
            print(f"🔍 [VERBOSE] Checking if {repr(recommended_ratio)} is in {valid_ratios}")

            # First check if ratio is generally valid
            if recommended_ratio not in valid_ratios:
                print(f"⚠️ [VERBOSE] Invalid ratio {repr(recommended_ratio)}, using fallback '1:1'")
                recommended_ratio = "1:1"
            else:
                print(f"✅ [VERBOSE] Valid ratio {repr(recommended_ratio)} found!")

                # Then check scene-specific rules
                if scene_type and scene_type in scene_ratio_rules:
                    allowed_ratios = scene_ratio_rules[scene_type]
                    if recommended_ratio not in allowed_ratios:
                        print(f"⚠️ [VERBOSE] Ratio {repr(recommended_ratio)} not allowed for scene {repr(scene_type)}")
                        print(f"⚠️ [VERBOSE] Allowed ratios for {scene_type}: {allowed_ratios}")
                        print(f"🔄 [VERBOSE] Re-analyzing image to get correct ratio...")
                        # Re-analyze with stricter prompt focusing on allowed ratios
                        return self._reanalyze_with_constraints(scene_type, allowed_ratios, crop_focus, reasoning, ollama_response)
                    else:
                        print(f"✅ [VERBOSE] Ratio {repr(recommended_ratio)} is valid for scene {repr(scene_type)}")

            # Clean up other fields too
            if scene_type:
                scene_type = scene_type.strip().lower()
                # Normalize scene type names
                if scene_type in ['cockpit_interior', 'interior', 'cockpit']:
                    scene_type = 'cockpit_interior'
                elif scene_type in ['detail', 'detail_shot']:
                    scene_type = 'detail'
                elif scene_type in ['parked', 'parked_shot']:
                    scene_type = 'parked'
                elif scene_type in ['driving_distant', 'distant', 'driving']:
                    scene_type = 'driving_distant'
                elif scene_type in ['dynamic_driving', 'dynamic', 'action']:
                    scene_type = 'dynamic_driving'
                print(f"🔍 [VERBOSE] Normalized scene_type: {scene_type}")
            if crop_focus:
                crop_focus = crop_focus.strip()
            if reasoning:
                reasoning = reasoning.strip()

            return scene_type, recommended_ratio, crop_focus, reasoning

        except Exception as e:
            print(f"❌ [VERBOSE] Exception during parsing: {e}")
            import traceback
            print(f"❌ [VERBOSE] Full traceback:\n{traceback.format_exc()}")
            # Fallback to default values
            return "unknown", "1:1", "center of image", f"Failed to parse Ollama response: {e}"

    def _reanalyze_with_constraints(self, scene_type, allowed_ratios, crop_focus, reasoning, ollama_response):
        """Re-analyze with specific ratio constraints for the detected scene type"""
        print(f"🔄 [VERBOSE] Re-analyzing for scene {scene_type} with allowed ratios: {allowed_ratios}")

        if scene_type == 'driving_distant':
            # For driving distant, prefer 4:5 for balanced composition, then 16:9 for landscape
            if '4:5' in allowed_ratios:
                selected_ratio = '4:5'
            elif '16:9' in allowed_ratios:
                selected_ratio = '16:9'
            else:
                selected_ratio = allowed_ratios[0]
        elif scene_type == 'cockpit_interior':
            # For cockpit interior, use intelligent analysis
            cockpit_analysis = self.analyze_cockpit_layout(ollama_response)
            context_info = {'car_coverage': 0.2}  # Default context
            selected_ratio, confidence = self.intelligent_cockpit_ratio_selection(cockpit_analysis, context_info)

            # Ensure selected ratio is in allowed ratios
            if selected_ratio not in allowed_ratios:
                selected_ratio = allowed_ratios[0]
                print(f"🔄 [VERBOSE] Intelligent selection {selected_ratio} not in allowed ratios, using {allowed_ratios[0]}")
        else:
            # For other scenes, use first allowed ratio
            selected_ratio = allowed_ratios[0]

        print(f"✅ [VERBOSE] Selected ratio after re-analysis: {selected_ratio}")
        return scene_type, selected_ratio, crop_focus, reasoning

    def analyze_image_context(self, yolo_results, img_width, img_height):
        """Analyze YOLO detection results to provide context for scene classification."""
        context_info = {
            'car_coverage': 0.0,  # Percentage of image covered by cars
            'car_count': 0,
            'largest_car_size': 0.0,
            'has_person': False,
            'scene_complexity': 'simple'  # simple, moderate, complex
        }

        if not yolo_results or len(yolo_results[0].boxes) == 0:
            return context_info

        boxes = yolo_results[0].boxes.xywh
        classes = yolo_results[0].boxes.cls
        class_names = yolo_results[0].names

        total_image_area = img_width * img_height
        car_areas = []

        for box, cls_idx in zip(boxes, classes):
            class_name = class_names[int(cls_idx)]
            x, y, w, h = box.cpu().numpy()
            box_area = w * h

            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                car_areas.append(box_area)
                context_info['car_count'] += 1
            elif class_name == 'person':
                context_info['has_person'] = True

        if car_areas:
            context_info['largest_car_size'] = max(car_areas) / total_image_area
            context_info['car_coverage'] = sum(car_areas) / total_image_area

        # Determine scene complexity
        total_objects = len(boxes)
        if total_objects <= 2:
            context_info['scene_complexity'] = 'simple'
        elif total_objects <= 5:
            context_info['scene_complexity'] = 'moderate'
        else:
            context_info['scene_complexity'] = 'complex'

        return context_info

    def get_crop_focus_point(self, img, crop_focus_description, yolo_results):
        """Determine the optimal crop center point based on Ollama analysis and YOLO detection."""
        width, height = img.size
        center_x, center_y = width / 2, height / 2  # Default center

        # Use YOLO results to find relevant objects
        if yolo_results and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes.xywh
            classes = yolo_results[0].boxes.cls
            class_names = yolo_results[0].names

            # Look for cars first
            car_boxes = []
            for box, cls_idx in zip(boxes, classes):
                class_name = class_names[int(cls_idx)]
                if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                    car_boxes.append(box)

            if car_boxes:
                # Use the largest car as the focus point
                largest_car = max(car_boxes, key=lambda b: b[2] * b[3])
                x, y, _, _ = largest_car.cpu().numpy()  # Only use x, y coordinates
                center_x, center_y = x, y
            else:
                # If no cars found, use the largest detected object
                largest_box = max(boxes, key=lambda b: b[2] * b[3])
                x, y, _, _ = largest_box.cpu().numpy()  # Only use x, y coordinates
                center_x, center_y = x, y

        # Parse crop focus description for specific instructions
        if crop_focus_description and isinstance(crop_focus_description, str):
            focus_lower = crop_focus_description.lower()
            # Adjust focus based on description
            if 'top' in focus_lower or 'upper' in focus_lower:
                center_y = center_y * 0.7  # Move focus towards top
            elif 'bottom' in focus_lower or 'lower' in focus_lower:
                center_y = center_y * 1.3  # Move focus towards bottom
            elif 'left' in focus_lower:
                center_x = center_x * 0.7  # Move focus towards left
            elif 'right' in focus_lower:
                center_x = center_x * 1.3  # Move focus towards right

        return center_x, center_y

    def crop(self, image, prompt, model, ollama_host="http://localhost:11434", enable_reasoning=True, temperature=0.1):
        # Note: prompt is kept for interface compatibility but Ollama analysis takes precedence
        # 1. Convert the input tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # 2. Encode the image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 3. YOLO Object Detection
        yolo_results = self.model.predict(img)

        # 4. Ollama Scene Analysis and Aspect Ratio Decision
        ollama_analysis = ""
        recommended_ratio = "1:1"  # Default fallback
        scene_type = "unknown"
        crop_focus = "center of image"
        reasoning = "Using default 1:1 ratio"

        if enable_reasoning:
            # Add image context analysis
            context_info = self.analyze_image_context(yolo_results, img.width, img.height)
            print(f"🔍 [CONTEXT] Car coverage: {context_info['car_coverage']:.2%}")
            print(f"🔍 [CONTEXT] Largest car size: {context_info['largest_car_size']:.2%}")
            print(f"🔍 [CONTEXT] Car count: {context_info['car_count']}")
            print(f"🔍 [CONTEXT] Scene complexity: {context_info['scene_complexity']}")

            ollama_analysis = self.analyze_scene_with_ollama(img_base64, model, ollama_host, temperature)
            scene_type, recommended_ratio, crop_focus, reasoning = self.parse_ollama_analysis(ollama_analysis)

            # Enhanced cockpit analysis for intelligent ratio selection
            if scene_type == 'cockpit_interior':
                print(f"🧠 [COCKPIT] Detected cockpit scene, performing intelligent ratio analysis...")
                cockpit_analysis = self.analyze_cockpit_layout(ollama_analysis)
                intelligent_ratio, confidence = self.intelligent_cockpit_ratio_selection(cockpit_analysis, context_info)

                # Override the recommended ratio with intelligent selection
                if intelligent_ratio in ['4:5', '1:1']:  # Ensure it's a valid cockpit ratio
                    recommended_ratio = intelligent_ratio
                    print(f"🧠 [COCKPIT] Intelligent selection: {intelligent_ratio} (confidence: {confidence})")
                else:
                    print(f"⚠️ [COCKPIT] Intelligent selection failed, keeping original: {recommended_ratio}")

            # Additional validation for interior vs detail confusion
            if scene_type == 'detail' and context_info['car_coverage'] < 0.1:
                print(f"⚠️ [VALIDATION] Scene classified as 'detail' but low car coverage ({context_info['car_coverage']:.2%})")
                print(f"⚠️ [VALIDATION] This might be an interior shot misclassified as detail")
            elif scene_type == 'cockpit_interior':
                print(f"✅ [VALIDATION] Interior scene correctly identified with intelligent ratio selection")

            print(f"🎯 [FINAL] Scene: {scene_type}, Ratio: {recommended_ratio}, Focus: {crop_focus}")

        # 5. Get the crop focus point
        center_x, center_y = self.get_crop_focus_point(img, crop_focus, yolo_results)

        # 6. Convert ratio string to numeric value
        ratio_map = {"16:9": 16/9, "9:16": 9/16, "1:1": 1.0, "4:5": 4/5}
        aspect_ratio = ratio_map[recommended_ratio]

        # 7. Perform non-destructive crop using PIL
        width, height = img.size
        image_aspect_ratio = float(width) / height

        if image_aspect_ratio > aspect_ratio:
            # Image is wider than target, crop horizontally
            new_height = height
            new_width = int(round(height * aspect_ratio))
        else:
            # Image is taller or equal to target, crop vertically
            new_width = width
            new_height = int(round(width / aspect_ratio))

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

        # Perform the non-destructive crop using PIL
        cropped_img = img.crop((left, top, right, bottom))

        # 8. Convert the cropped PIL Image back to a tensor
        output_image = np.array(cropped_img).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image).unsqueeze(0)

        # 9. Prepare detailed reasoning output
        detailed_reasoning = f"Scene Analysis: {scene_type}\nRecommended Ratio: {recommended_ratio}\nCrop Focus: {crop_focus}\nReasoning: {reasoning}"

        return (output_image, detailed_reasoning, ollama_analysis)
