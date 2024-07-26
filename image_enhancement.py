# image_enhancement.py

import logging
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import cv2
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionControlNetPipeline, ControlNetModel
from torchvision import transforms
from config import (
    LOG_FORMAT, LOG_DATE_FORMAT, UPSCALER_MODEL, SD_BASE_MODEL,
    CONTROLNET_MODEL, CONTROLNET_CONDITIONING_SCALE,
    CONTROL_GUIDANCE_START, CONTROL_GUIDANCE_END
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def latent_upscale_image(image: Image.Image, prompt: str, output_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Upscale the given image using Stable Diffusion x2 latent upscaler.
    
    Args:
        image: PIL Image to upscale.
        prompt: Text prompt for guided upscaling.
        output_size: Desired output size as a tuple (width, height).
    
    Returns:
        np.ndarray: Upscaled image as a numpy array.
    """
    try:
        sd_pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(UPSCALER_MODEL, torch_dtype=dtype)
        sd_pipeline.to(device)
        
        # Upscale the image
        with torch.no_grad():
            upscaled_image = sd_pipeline(
                prompt=prompt,
                image=image.resize((512, 512)),  # Resize to 512x512 as required by the upscaler
                num_inference_steps=20,
                guidance_scale=0,
            ).images[0]
        
        upscaled_image = upscaled_image.resize(output_size, Image.LANCZOS)
        
        logger.info(f"Image upscaled successfully to {output_size[0]}x{output_size[1]}")
        return np.array(upscaled_image)
    except Exception as e:
        logger.error(f"Error during latent image upscaling: {str(e)}")
        logger.info("Falling back to simple resize")
        return np.array(image.resize(output_size, Image.LANCZOS))

def apply_freestyle(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply Freestyle to the given image.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for Freestyle.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    # TODO: Implement Freestyle functionality
    logger.info("Freestyle functionality not yet implemented")
    return image

def apply_controlnet(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply ControlNet to the given image and output at 1024x1024 resolution.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for ControlNet.
    
    Returns:
        np.ndarray: Processed image as a numpy array at 1024x1024 resolution.
    """
    try:
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
        
        # Load Stable Diffusion pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype
        )
        pipe.to(device)
        
        # Prepare control image (Canny edge detection)
        control_image = get_canny_image(image)
        
        # Resize control image to 1024x1024
        control_image = control_image.resize((1024, 1024), Image.LANCZOS)
        
        # Generate image
        output = pipe(
            prompt,
            image=control_image,
            num_inference_steps=10,  # Increased for better quality
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
            control_guidance_start=CONTROL_GUIDANCE_START,
            control_guidance_end=CONTROL_GUIDANCE_END,
            height=1024,
            width=1024
        ).images[0]
        
        logger.info("ControlNet processing completed successfully at 1024x1024 resolution")
        return np.array(output)
    except Exception as e:
        logger.error(f"Error during ControlNet processing: {str(e)}")
        return cv2.resize(image, (1024, 1024))  # Fallback to simple resize

def get_canny_image(image: np.ndarray) -> Image.Image:
    """
    Apply Canny edge detection to the input image.
    
    Args:
        image: Input image as a numpy array.
    
    Returns:
        PIL.Image.Image: Canny edge detected image.
    """
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)