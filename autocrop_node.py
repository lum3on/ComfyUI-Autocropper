import torch
from PIL import Image
import numpy as np
import base64
import io
import openai
import os
import cv2
import re
from dotenv import load_dotenv
from ultralytics import YOLO
import requests

# Load environment variables from .env file, overriding any existing ones
load_dotenv(override=True)

class AutoCropperNode:
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
    def get_models(cls):
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            models = client.models.list()
            # Filter for models that are compatible with the vision API
            vision_models = [m.id for m in models if "vision" in m.id or "o" in m.id]
            return vision_models
        except Exception as e:
            print(f"Could not fetch OpenAI models: {e}")
            return ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "This automotive-focused node uses GPT to analyze car images and intelligently crop them based on scene type. It follows specific rules: Exterior shots (1:1/4:5), Driving shots where car is distant (9:16/4:5), Parked shots (16:9/1:1), Cockpit/interior views (4:5/1:1), Detail shots (1:1/4:5), Dynamic driving scenes (16:9/9:16). The crop is non-destructive and preserves image quality."
                }),
                "model": (s.get_models(),),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_reasoning": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("cropped_image", "reasoning", "raw_api_response")

    FUNCTION = "crop"

    CATEGORY = "Image"

    def analyze_scene_with_gpt(self, img_base64, model, seed=0):
        """Use GPT to analyze the scene and determine the best aspect ratio based on automotive cropping rules."""
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            scene_analysis_prompt = """Analyze this automotive image and determine the best aspect ratio for cropping based on these specific rules:

CROPPING RULES:
• Exterior shots: 1:1 or 4:5
• Driving shots where the car is not very close: 9:16 or 4:5
• Parked shots: 16:9 or 1:1
• Cockpit/interior views: 4:5 or 1:1
• Detail shots (wheels, badges, headlights, stitching): 1:1 or 4:5
• Dynamic driving scenes (e.g. cornering, acceleration): 16:9 or 9:16

AVAILABLE ASPECT RATIOS: 16:9, 9:16, 1:1, 4:5

Please respond in this exact format:
SCENE_TYPE: [one of: exterior, driving_distant, parked, cockpit_interior, detail, dynamic_driving]
RECOMMENDED_RATIO: [one of: 16:9, 9:16, 1:1, 4:5]
CROP_FOCUS: [describe the main subject/area that should be the focus of the crop]
REASONING: [brief explanation of why this ratio and focus point work best for this scene type]"""

            api_params = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": scene_analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 400,
            }

            if seed > 0:
                api_params["seed"] = seed
                api_params["temperature"] = 0.1  # Low temperature for consistent analysis

            response = client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            return f"GPT analysis failed: {e}"

    def parse_gpt_analysis(self, gpt_response):
        """Parse the GPT response to extract scene type, recommended ratio, and reasoning."""
        try:
            lines = gpt_response.strip().split('\n')
            scene_type = None
            recommended_ratio = None
            crop_focus = None
            reasoning = None

            for line in lines:
                if line.startswith('SCENE_TYPE:'):
                    scene_type = line.split(':', 1)[1].strip()
                elif line.startswith('RECOMMENDED_RATIO:'):
                    recommended_ratio = line.split(':', 1)[1].strip()
                elif line.startswith('CROP_FOCUS:'):
                    crop_focus = line.split(':', 1)[1].strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()

            # Validate the recommended ratio
            valid_ratios = ["16:9", "9:16", "1:1", "4:5"]
            if recommended_ratio not in valid_ratios:
                recommended_ratio = "1:1"  # Default fallback

            return scene_type, recommended_ratio, crop_focus, reasoning

        except Exception as e:
            # Fallback to default values
            return "exterior", "1:1", "center of image", f"Failed to parse GPT response: {e}"

    def get_crop_focus_point(self, img, crop_focus_description, yolo_results):
        """Determine the optimal crop center point based on GPT analysis and YOLO detection."""
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

        # Note: crop_focus_description could be used for future enhancements
        # to parse specific focus instructions from GPT

        return center_x, center_y

    def crop(self, image, prompt, model, seed=0, enable_reasoning=True):
        # Note: prompt is kept for interface compatibility but GPT analysis takes precedence
        # 1. Convert the input tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # 2. Encode the image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 3. YOLO Object Detection
        yolo_results = self.model.predict(img)

        # 4. GPT Scene Analysis and Aspect Ratio Decision
        gpt_analysis = ""
        recommended_ratio = "1:1"  # Default fallback
        scene_type = "exterior"
        crop_focus = "center of image"
        reasoning = "Using default 1:1 ratio"

        if enable_reasoning:
            gpt_analysis = self.analyze_scene_with_gpt(img_base64, model, seed)
            scene_type, recommended_ratio, crop_focus, reasoning = self.parse_gpt_analysis(gpt_analysis)

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

        return (output_image, detailed_reasoning, gpt_analysis)
