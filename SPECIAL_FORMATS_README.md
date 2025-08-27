# 🎯 Special Formats Auto Cropper

An intelligent ComfyUI node that automatically analyzes advertising content and crops images to exact pixel dimensions for standard advertising banner formats.

## ✨ Features

- **🤖 Automatic Format Selection**: Uses Ollama/Qwen vision models to analyze content and automatically select the best advertising format
- **📐 Exact Pixel Dimensions**: Outputs precise pixel dimensions required by advertising platforms
- **🎨 Advertising-Focused Analysis**: Specialized for products, logos, text, and brand elements
- **🔍 Smart Object Detection**: Enhanced YOLO detection for advertising content
- **📱 5 Standard Formats**: Supports all major advertising banner formats

## 📊 Supported Advertising Formats

| Format | Dimensions | Aspect Ratio | Use Case |
|--------|------------|--------------|----------|
| Desktop Banner | 468×60 | 7.8:1 | Standard web banner advertising |
| Desktop Leaderboard | 728×90 | 8.09:1 | Top-of-page banner advertising |
| Desktop Skyscraper | 120×600 | 0.2:1 | Sidebar vertical advertising |
| Desktop Large Skyscraper | 160×600 | 0.27:1 | Wide sidebar vertical advertising |
| Mobile Leaderboard | 320×50 | 6.4:1 | Mobile banner advertising |

## 🚀 How It Works

1. **Content Analysis**: Analyzes the input image for advertising content types:
   - Product-focused images
   - Brand/logo content
   - Text-heavy layouts
   - Mixed content
   - Mobile-optimized designs

2. **Automatic Format Selection**: Based on content analysis, automatically selects the optimal advertising format:
   - Horizontal compositions → Banner/Leaderboard formats
   - Vertical compositions → Skyscraper formats
   - Product images → Format based on product orientation
   - Text-heavy content → Leaderboard for readability
   - Mobile content → Mobile Leaderboard

3. **Intelligent Cropping**: Uses YOLO object detection and vision model guidance to:
   - Focus on main products/subjects
   - Preserve important text and branding
   - Maintain visual hierarchy
   - Ensure key elements remain visible

4. **Exact Pixel Output**: Crops and resizes to exact pixel dimensions with validation

## 🛠️ Requirements

- **Ollama**: Must be running locally with a vision-capable model
- **Recommended Models**: 
  - `qwen2.5vl:7b` (best performance)
  - `qwen2.5vl:latest`
  - `llama3.2-vision:latest`
  - `llava:7b`

## 📝 Usage

1. **Load Image**: Connect any image input to the node
2. **Select Model**: Choose from available Ollama vision models
3. **Configure Settings** (optional):
   - Adjust temperature for analysis consistency
   - Enable/disable reasoning output
   - Set custom Ollama host if needed
4. **Run**: The node will automatically analyze and crop to the best format

## 🔧 Outputs

- **Cropped Image**: Exact pixel dimensions for the selected format
- **Format Info**: Details about the selected format and dimensions
- **Reasoning**: Analysis explanation and crop focus details
- **Raw Analysis**: Full Ollama model response for debugging

## 🎯 Content Type Detection

The node intelligently detects and handles different advertising content:

- **Product-Focused**: Single products, clean backgrounds
- **Brand-Focused**: Logos, branding elements, corporate content
- **Text-Heavy**: Headlines, copy, multiple text elements
- **Mixed Content**: Products + text + branding combined
- **Mobile-Optimized**: Content suitable for mobile displays

## 🔍 Smart Format Selection Rules

- **Wide horizontal layouts** → Desktop Banner or Leaderboard
- **Tall vertical layouts** → Desktop Skyscraper formats
- **Product images** → Format based on product orientation
- **Logo/branding** → Banner or Leaderboard for visibility
- **Text-heavy content** → Leaderboard for readability
- **Mobile content** → Mobile Leaderboard format

## 🚨 Important Notes

- Requires Ollama to be running locally
- Vision-capable models are required for image analysis
- Output images are exactly the specified pixel dimensions
- Original image quality is preserved during cropping
- Works best with marketing/advertising content

## 🔗 Integration

The node appears in ComfyUI under the **Image** category as "🎯 Special Formats Auto Cropper" and integrates seamlessly with existing workflows.

## 🧪 Testing

Run the included test script to verify functionality:
```bash
python test_special_formats.py
```

## 📈 Performance

- **Analysis Time**: 2-5 seconds depending on model and image size
- **Accuracy**: High precision for advertising content detection
- **Memory Usage**: Efficient with automatic cleanup
- **Compatibility**: Works with all ComfyUI image workflows
