# ComfyUI AutoCropper & Image Utilities

This repository contains a suite of custom nodes for ComfyUI designed to automate image cropping and provide additional image format support. The nodes leverage object detection and generative AI to intelligently analyze and crop images, along with utilities for handling TIFF files.

## Features

- **AI-Powered Cropping**: Utilizes YOLO for object detection and generative models (OpenAI GPT-4o, Ollama) to determine the optimal crop for an image based on its content and composition.
- **Multiple Cropping Nodes**:
    - **Auto Crop Porsche**: A specialized node fine-tuned for automotive imagery using OpenAI's models.
    - **Ollama Auto Cropper**: A flexible cropping node that uses local Ollama models for analysis, offering privacy and customization.
- **TIFF Image Support**:
    - **Load TIFF Image**: A node to load high-resolution TIFF images, including various bit depths, into ComfyUI.
    - **Save TIFF Image**: A node to save tensors as TIFF files with compression options.

## Installation

1.  **Clone the Repository**:
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```bash
    git clone <URL_of_this_repository> ComfyUI_AutoCropper
    ```

2.  **Install Dependencies**:
    The required Python packages will be installed automatically when ComfyUI starts. You can also install them manually by running:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `Pillow`, `ultralytics`, `opencv-python-headless`, `ollama`, and other necessary libraries.

3.  **Download Models**:
    - The required **YOLOv8s** model for object detection will be downloaded automatically on the first run and saved to `ComfyUI/models/ultralytics/`.
    - For the **Ollama Auto Cropper** node, ensure you have Ollama installed and have pulled a vision-capable model (e.g., `ollama pull llava`).

4.  **Restart ComfyUI**:
    After installation, restart ComfyUI to load the new nodes.

## Nodes Included

### Image Cropping

#### 1. Auto Crop Porsche

-   **Category**: `Image`
-   **Description**: This node uses an OpenAI model (like GPT-4o) to analyze automotive images and suggest the best crop based on predefined rules for different scene types (e.g., exterior shots, driving scenes, interior views).
-   **Inputs**:
    - `image`: The input image to crop.
    - `prompt`: A text field (pre-filled with rules) to guide the AI.
    - `model`: A dropdown to select the OpenAI model.
    - `seed` (optional): For reproducible results.
    - `enable_reasoning` (optional): Toggle to enable/disable AI analysis.
-   **Outputs**:
    - `cropped_image`: The resulting cropped image.
    - `reasoning`: A string explaining the AI's decision.
    - `raw_api_response`: The full response from the OpenAI API for debugging.

#### 2. Ollama Auto Cropper

-   **Category**: `Image`
-   **Description**: Similar to the OpenAI version, but uses a local Ollama instance for analysis. This is ideal for users who want to use local models for privacy or cost reasons. It features an advanced, rule-based prompt to guide the local model in making precise cropping decisions.
-   **Inputs**:
    - `image`: The input image.
    - `prompt`: A detailed, pre-filled prompt with a decision tree for the Ollama model.
    - `model`: A dropdown of available local Ollama vision models.
    - `ollama_host` (optional): The URL for your Ollama server.
    - `enable_reasoning` (optional): Toggle AI analysis.
    - `temperature` (optional): Controls the creativity of the Ollama model.
-   **Outputs**:
    - `cropped_image`: The cropped image.
    - `reasoning`: The explanation from the Ollama model.
    - `raw_api_response`: The full response from the Ollama API.

### Image File I/O

#### 3. Load TIFF Image

-   **Category**: `image`
-   **Description**: Loads a TIFF (`.tif`) image from a file path. It handles various bit depths and converts the image into a standard tensor that ComfyUI can use.
-   **Inputs**:
    - `image`: The path to the TIFF image file.
-   **Outputs**:
    - `image`: The loaded image as an `IMAGE` tensor.
    - `mask`: The alpha channel of the image as a `MASK`.

#### 4. Save TIFF Image

-   **Category**: `image`
-   **Description**: Saves an `IMAGE` tensor to a TIFF file.
-   **Inputs**:
    - `images`: The images to save.
    - `filename_prefix`: The prefix for the output filename.
    - `compression`: The compression method to use (`none`, `tiff_lzw`, `tiff_deflate`).
-   **Outputs**: This is an output node and saves files to the ComfyUI `output` directory.

## Dependencies

- `Pillow`
- `numpy`
- `torch`
- `ultralytics`
- `opencv-python-headless`
- `ollama`
- `openai`
- `python-dotenv`
- `requests`
- `smartcrop` (from git)

These are listed in `requirements.txt` and should be handled automatically.