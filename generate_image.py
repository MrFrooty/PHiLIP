# generate_image.py

import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from torchvision import transforms
from PIL import Image
import logging
from typing import List, Optional, Union
from tqdm import tqdm
from config import MODEL_MID_RES, MODEL_HIGH_RES, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# Preprocessing pipeline for images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained pipelines
pipe_mid_res = PixArtAlphaPipeline.from_pretrained(MODEL_MID_RES)
pipe_high_res = PixArtAlphaPipeline.from_pretrained(MODEL_HIGH_RES)

def generate_images(prompt: str, num_images: int = 1, resolution: int = 512, temp: float = 0.7, 
                    base_images: Optional[Union[List[np.ndarray], np.ndarray]] = None, steps: int = 50) -> List[np.ndarray]:
    """
    Generate images based on the given prompt and parameters.
    
    Args:
        prompt: The text prompt for image generation.
        num_images: Number of images to generate.
        resolution: Image resolution (512 or 1024).
        temp: Temperature for generation guidance.
        base_images: Base images for generation.
        steps: Number of inference steps.
    
    Returns:
        List[np.ndarray]: List of generated images as numpy arrays.
    """
    pipe = pipe_mid_res if resolution == 512 else pipe_high_res
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    input_images = process_base_images(base_images, device) if base_images else None

    try:
        with torch.no_grad():
            results = pipe(
                [prompt] * num_images,
                num_images_per_prompt=1,
                height=resolution,
                width=resolution,
                guidance_scale=temp,
                image=input_images,
                num_inference_steps=steps,
                callback=lambda i, t, x: tqdm.write(f"Step {i}/{steps}", end="\r")
            )
        logger.info(f"Generated {num_images} images successfully")
        return [np.array(image) for image in results.images]
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return []

def process_base_images(base_images: Union[List[np.ndarray], np.ndarray], device: str) -> torch.Tensor:
    """
    Process base images for use in generation.
    
    Args:
        base_images: Base images to process.
        device: Device to use for processing.
    
    Returns:
        torch.Tensor: Processed base images as a tensor.
    """
    if isinstance(base_images, list):
        base_image_tensors = [preprocess(Image.fromarray(img) if isinstance(img, np.ndarray) else img).unsqueeze(0) for img in base_images]
        return torch.mean(torch.stack(base_image_tensors), dim=0).to(device)
    else:
        return preprocess(Image.fromarray(base_images) if isinstance(base_images, np.ndarray) else base_images).unsqueeze(0).to(device)