# generate_image.py

import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from torchvision import transforms
from PIL import Image
import logging
from typing import List, Optional, Union
from config import MODEL_MID_RES, MODEL_HIGH_RES, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Determine and return the appropriate device for computation.

    Returns:
        torch.device: The device to be used for computation (CUDA or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        if 'AMD' in device_name or 'MI' in device_name:
            logger.info(f"AMD GPU detected. ROCm version: {torch.version.hip}")
            torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logger.info("GPU not available. Using CPU")
    return device

device = get_device()

dtype = torch.float16 if device.type == "cuda" else torch.float32

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pipe_mid_res = PixArtAlphaPipeline.from_pretrained(MODEL_MID_RES, torch_dtype=dtype)
pipe_high_res = PixArtAlphaPipeline.from_pretrained(MODEL_HIGH_RES, torch_dtype=dtype)

def generate_images(prompt: str, num_images: int = 1, resolution: int = 512, temp: float = 0.7, 
                    base_images: Optional[Union[List[np.ndarray], np.ndarray]] = None, steps: int = 10) -> List[np.ndarray]:
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
    pipe = pipe_high_res if resolution == 1024 else pipe_mid_res
    pipe.to(device)

    input_images = process_base_images(base_images, resolution) if base_images else None

    try:
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    results = generate_with_pipeline(pipe, prompt, num_images, resolution, temp, input_images, steps)
            else:
                results = generate_with_pipeline(pipe, prompt, num_images, resolution, temp, input_images, steps)
        logger.info(f"Generated {num_images} images successfully")
        return [np.array(image) for image in results.images]
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return []

def process_base_images(base_images: Union[List[np.ndarray], np.ndarray], target_resolution: int) -> torch.Tensor:
    """
    Process base images for use in generation.
    
    Args:
        base_images: Base images to process.
        target_resolution: Target resolution for the processed images.
    
    Returns:
        torch.Tensor: Processed base images as a tensor.
    """
    if isinstance(base_images, np.ndarray):
        base_images = [base_images]
    
    processed_images = []
    for img in base_images:
        pil_img = Image.fromarray(img).convert("RGB").resize((target_resolution, target_resolution))
        tensor_img = preprocess(pil_img).unsqueeze(0)
        processed_images.append(tensor_img)
    
    return torch.cat(processed_images, dim=0).to(device)

def generate_with_pipeline(pipe, prompt, num_images, resolution, temp, input_images, steps):
    """
    Helper function to generate images with the pipeline.

    Args:
        pipe: The PixArtAlphaPipeline to use for generation.
        prompt: The text prompt for image generation.
        num_images: Number of images to generate.
        resolution: Image resolution.
        temp: Temperature for generation guidance.
        input_images: Processed input images.
        steps: Number of inference steps.

    Returns:
        The results of the image generation.
    """
    return pipe(
        [prompt] * num_images,
        num_images_per_prompt=1,
        height=resolution,
        width=resolution,
        guidance_scale=temp,
        image=input_images,
        num_inference_steps=steps
    )

logger.info(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")