# image_enhancement.py

import logging
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
import cv2
import json
import re
from diffusers import DiffusionPipeline, StableDiffusionLatentUpscalePipeline, StableDiffusionControlNetPipeline, ControlNetModel
from torchvision import transforms
from config import (
    LOG_FORMAT, LOG_DATE_FORMAT, UPSCALER_MODEL, SD_BASE_MODEL,
    CONTROLNET_MODEL, CONTROLNET_CONDITIONING_SCALE,
    CONTROL_GUIDANCE_START, CONTROL_GUIDANCE_END,
    FREESTYLE_PROMPT_JSON, FREESTYLE_N, FREESTYLE_B, FREESTYLE_S,
    MODEL_HIGH_RES, FREESTYLE_MODEL
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Determine and return the appropriate device for computation."""
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

# Use mixed precision for GPU, full precision for CPU
dtype = torch.float16 if device.type == "cuda" else torch.float32

def load_style_prompts() -> List[Dict[str, str]]:
    """Load and return the style prompts from the JSON file."""
    try:
        with open(FREESTYLE_PROMPT_JSON, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading style prompts: {str(e)}")
        return []

def select_style_prompt(style_prompts: List[Dict[str, str]]) -> Dict[str, str]:
    """Prompt the user to select a style from the available options."""
    print("Available styles:")
    for i, style in enumerate(style_prompts):
        print(f"{i + 1}. {style['name']}")
    
    while True:
        try:
            choice = int(input("Select a style by number: ")) - 1
            if 0 <= choice < len(style_prompts):
                return style_prompts[choice]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def combine_prompts(user_prompt: str, style: Dict[str, str], max_tokens: int = 77) -> Tuple[str, str]:
    """
    Combine user prompt with style prompt, ensuring it doesn't exceed the token limit.
    Truncates the negative prompt first, then the positive prompt if necessary.
    
    Args:
        user_prompt: The user's input prompt.
        style: The selected style dictionary.
        max_tokens: Maximum number of tokens allowed (default is 77 for SDXL).
    
    Returns:
        Tuple[str, str]: Combined positive prompt and negative prompt.
    """
    def count_tokens(text: str) -> int:
        # This is a simple approximation. For more accurate counting, consider using a tokenizer.
        return len(re.findall(r'\w+', text))

    user_tokens = count_tokens(user_prompt)
    style_tokens = count_tokens(style['prompt'])
    neg_tokens = count_tokens(style.get('negative_prompt', ''))

    # Combine user prompt and style prompt
    full_prompt = f"{user_prompt}, {style['prompt']}"
    full_prompt_tokens = user_tokens + style_tokens

    # Calculate remaining tokens for negative prompt
    remaining_tokens = max_tokens - full_prompt_tokens

    # Truncate negative prompt if necessary
    if neg_tokens > remaining_tokens:
        words = style['negative_prompt'].split()
        neg_prompt = ' '.join(words[:remaining_tokens])
        logger.warning(f"Negative prompt truncated to fit within token limit. Original: {style['negative_prompt']}, Truncated: {neg_prompt}")
    else:
        neg_prompt = style.get('negative_prompt', '')

    # If negative prompt is completely removed and we still exceed the token limit, truncate the positive prompt
    if not neg_prompt and full_prompt_tokens > max_tokens:
        words = full_prompt.split()
        full_prompt = ' '.join(words[:max_tokens])
        logger.warning(f"Positive prompt truncated to fit within token limit. Truncated: {full_prompt}")

    # Log warnings
    if not neg_prompt:
        logger.warning("Negative prompt completely removed due to token limit.")
    if full_prompt_tokens > max_tokens:
        logger.warning("Positive prompt truncated due to token limit.")

    return full_prompt, neg_prompt

def apply_freestyle(image: np.ndarray, prompt: str) -> np.ndarray:
    """
    Apply Freestyle to the given image using the SDXL model from Hugging Face.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for Freestyle.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    try:
        pipe = DiffusionPipeline.from_pretrained(FREESTYLE_MODEL, torch_dtype=dtype)
        pipe.to(device)

        style_prompts = load_style_prompts()
        if not style_prompts:
            raise ValueError("No style prompts available.")

        selected_style = select_style_prompt(style_prompts)
        
        # Combine prompts and handle potential token limit
        full_prompt, neg_prompt = combine_prompts(prompt, selected_style)

        input_image = Image.fromarray(image).resize((1024, 1024))

        with torch.no_grad():
            result = generate_image_with_pipeline(pipe, full_prompt, input_image, neg_prompt)

        logger.info(f"Freestyle processing completed successfully using {selected_style['name']} style")
        return np.array(result)
    except Exception as e:
        logger.error(f"Error during Freestyle processing: {str(e)}")
        logger.warning("Falling back to original image due to Freestyle processing error.")
        return image

def generate_image_with_pipeline(pipe, prompt: str, image: Image.Image, negative_prompt: str) -> Image.Image:
    """Generate image using the provided pipeline and parameters."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            result = pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=10,
                guidance_scale=5,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
            ).images[0]
    else:
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=10,
            guidance_scale=5,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
        ).images[0]
    return result

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
        
        with torch.no_grad():
            upscaled_image = generate_image_with_pipeline(sd_pipeline, prompt, image.resize((512, 512)), "")
        
        upscaled_image = upscaled_image.resize(output_size, Image.LANCZOS)
        
        logger.info(f"Image upscaled successfully to {output_size[0]}x{output_size[1]}")
        return np.array(upscaled_image)
    except Exception as e:
        logger.error(f"Error during latent image upscaling: {str(e)}")
        logger.info("Falling back to simple resize")
        return np.array(image.resize(output_size, Image.LANCZOS))

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
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype
        )
        pipe.to(device)
        
        control_image = get_canny_image(image).resize((1024, 1024), Image.LANCZOS)
        
        with torch.no_grad():
            output = generate_controlnet_image(pipe, prompt, control_image)
        
        logger.info("ControlNet processing completed successfully at 1024x1024 resolution")
        return np.array(output)
    except Exception as e:
        logger.error(f"Error during ControlNet processing: {str(e)}")
        return cv2.resize(image, (1024, 1024))  # Fallback to simple resize

def generate_controlnet_image(pipe, prompt: str, control_image: Image.Image) -> Image.Image:
    """Generate image using ControlNet pipeline."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = pipe(
                prompt,
                image=control_image,
                num_inference_steps=10,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                control_guidance_start=CONTROL_GUIDANCE_START,
                control_guidance_end=CONTROL_GUIDANCE_END,
                height=1024,
                width=1024
            ).images[0]
    else:
        output = pipe(
            prompt,
            image=control_image,
            num_inference_steps=10,
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
            control_guidance_start=CONTROL_GUIDANCE_START,
            control_guidance_end=CONTROL_GUIDANCE_END,
            height=1024,
            width=1024
        ).images[0]
    return output

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

def apply_pixart(image: np.ndarray, prompt: str, temperature: float) -> np.ndarray:
    """
    Apply Pixart enhancement to the given image.
    
    Args:
        image: Input image as a numpy array.
        prompt: Text prompt for Pixart.
        temperature: Temperature for generation guidance.
    
    Returns:
        np.ndarray: Processed image as a numpy array.
    """
    try:
        pipe = DiffusionPipeline.from_pretrained(MODEL_HIGH_RES, torch_dtype=dtype)
        pipe.to(device)

        input_image = Image.fromarray(image).resize((1024, 1024))

        with torch.no_grad():
            result = generate_pixart_image(pipe, prompt, input_image, temperature)

        logger.info("Pixart enhancement completed successfully")
        return np.array(result)
    except Exception as e:
        logger.error(f"Error during Pixart enhancement: {str(e)}")
        return image

def generate_pixart_image(pipe, prompt: str, image: Image.Image, temperature: float) -> Image.Image:
    """Generate image using Pixart pipeline."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            result = pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=10,
                guidance_scale=temperature,
                num_images_per_prompt=1,
            ).images[0]
    else:
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=10,
            guidance_scale=temperature,
            num_images_per_prompt=1,
        ).images[0]
    return result

def apply_enhancement(image: np.ndarray, prompt: str, enhancement_option: str, temperature: float = 5.0) -> np.ndarray:
    """Apply the selected enhancement to the image."""
    enhancement_functions = {
        "Freestyle": apply_freestyle,
        "Upscaler": lambda img, p: latent_upscale_image(Image.fromarray(img), p),
        "ControlNet": apply_controlnet,
        "Pixart": lambda img, p: apply_pixart(img, p, temperature),
        "None": lambda img, _: img
    }
    return enhancement_functions.get(enhancement_option, lambda img, _: img)(image, prompt)

# Print PyTorch and CUDA versions for debugging
logger.info(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")