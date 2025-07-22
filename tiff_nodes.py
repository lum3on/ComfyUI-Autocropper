import torch
import numpy as np
from PIL import Image, ImageOps
import os
import folder_paths

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class LoadTIFFImage:
    """
    A node to load a TIFF image from a file path.
    Handles various TIFF formats and bit depths, converting them to a standard RGB/RGBA tensor.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("STRING", {"default": "image.tif", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image: str):
        if not os.path.isabs(image):
            image_path = folder_paths.get_annotated_filepath(image)
        else:
            image_path = image

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Tiff image not found at: {image_path}")

        try:
            # Open the TIFF image
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)

            # Handle different modes and bit-depths
            if i.mode == 'I;16': # 16-bit grayscale
                # Convert to 8-bit grayscale for compatibility
                i = i.point(lambda x: x * (1./256.)).convert('L')

            image = i.convert("RGBA")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.mode:
                mask = image[:, :, :, 3]
                image = image[:, :, :, :3]
            else:
                mask = torch.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")

            return (image, mask)
        except Exception as e:
            print(f"Error loading TIFF image: {e}")
            # Return a dummy tensor on error to prevent workflow crash
            dummy_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (dummy_tensor, dummy_mask)


class SaveTIFFImage:
    """
    A node to save an image tensor as a TIFF file, with compression options.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "compression": (["none", "tiff_lzw", "tiff_deflate"],),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", compression="tiff_lzw", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            file = f"{filename}_{counter:05}_.tif"
            file_path = os.path.join(full_output_folder, file)
            
            try:
                # Save with specified compression
                img.save(file_path, compression=compression)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1
            except Exception as e:
                print(f"Error saving TIFF image: {e}")

        return {"ui": {"images": results}}

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadTIFFImage": LoadTIFFImage,
    "SaveTIFFImage": SaveTIFFImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTIFFImage": "Load TIFF Image",
    "SaveTIFFImage": "Save TIFF Image",
}