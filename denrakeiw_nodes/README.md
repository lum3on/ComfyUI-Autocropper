# Denrakeiw Nodes - ComfyUI Node Pack

A collection of custom nodes for ComfyUI that provide additional image generation and manipulation capabilities.

## Installation

1. Clone or download this repository to your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/denrakeiw_nodes/
   ```

2. Restart ComfyUI to load the new nodes.

## Nodes

### Color Generator Node

**Category:** `denrakeiw/image/generate`

Generates solid color images with customizable dimensions.

#### Features:
- 15 predefined colors to choose from
- Customizable width and height (1-4096 pixels)
- Returns image tensor, hex code, and color name
- Compatible with ComfyUI image processing pipeline

#### Available Colors:
- Red (#FF0000)
- Green (#00FF00)
- Blue (#0000FF)
- Yellow (#FFFF00)
- Cyan (#00FFFF)
- Magenta (#FF00FF)
- White (#FFFFFF)
- Black (#000000)
- Orange (#FFA500)
- Purple (#800080)
- Pink (#FFC0CB)
- Brown (#A52A2A)
- Gray (#808080)
- Navy (#000080)
- Lime (#00FF00)

#### Inputs:
- **color_name**: Select from the predefined color list (default: Red)
- **width**: Image width in pixels (1-4096, default: 512)
- **height**: Image height in pixels (1-4096, default: 512)

#### Outputs:
- **image**: Generated image tensor in ComfyUI format
- **hex_code**: Hexadecimal color code (e.g., "#FF0000")
- **color_name**: Name of the selected color

#### Usage:
1. Add the "Color Generator" node to your workflow
2. Select your desired color from the dropdown
3. Set the width and height for your image
4. Connect the image output to other nodes in your pipeline

## Requirements

- ComfyUI
- PyTorch
- PIL (Pillow)
- NumPy

## Version History

### v1.0.0
- Initial release
- Added Color Generator Node with 15 predefined colors
- Support for custom dimensions up to 4096x4096 pixels

## Author

Created by denrakeiw

## License

This project is open source. Please check the main repository for license details.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this node pack.

## Support

If you encounter any issues or have questions, please create an issue in the repository.
