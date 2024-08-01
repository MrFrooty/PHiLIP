#image_enhancement.py

import logging
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List, Callable
import torch
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline
import cv2
import json
from config import (
    UPSCALER_MODEL, SD_BASE_MODEL, CONTROLNET_MODEL, CONTROLNET_CONDITIONING_SCALE,
    CONTROL_GUIDANCE_START, CONTROL_GUIDANCE_END, FREESTYLE_MODEL, FREESTYLE_PROMPT_JSON
)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

def apply_enhancement(
    image: np.ndarray,
    prompt: str,
    enhancement_option: str,
    temperature: float = 1.0,
    selected_style: Optional[str] = None,
    output_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    enhancement_functions = {
        "Freestyle": apply_freestyle,
        "Upscaler": apply_upscaler,
        "ControlNet": apply_controlnet,
        "Pixart": apply_pixart
    }
    
    if enhancement_option not in enhancement_functions:
        logger.warning(f"Unknown enhancement option: {enhancement_option}. Returning original image.")
        return image
    
    try:
        return enhancement_functions[enhancement_option](
            image, prompt, temperature, selected_style, output_size
        )
    except Exception as e:
        logger.error(f"Error during {enhancement_option} enhancement: {str(e)}")
        return image

def apply_freestyle(
    image: np.ndarray,
    prompt: str,
    temperature: float,
    selected_style: Optional[str],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    pipe = DiffusionPipeline.from_pretrained(FREESTYLE_MODEL, torch_dtype=dtype).to(device)
    
    style_prompts = load_style_prompts()
    if not style_prompts:
        raise ValueError("No style prompts available.")
    
    selected_style_prompt = next((s for s in style_prompts if s['name'] == selected_style), None)
    if not selected_style_prompt:
        raise ValueError(f"Selected style '{selected_style}' not found.")
    
    full_prompt, neg_prompt = combine_prompts(prompt, selected_style_prompt)
    
    input_image = Image.fromarray(image).resize((1024, 1024))
    
    with torch.no_grad():
        result = pipe(
            prompt=full_prompt,
            image=input_image,
            negative_prompt=neg_prompt,
            num_inference_steps=20,
            guidance_scale=temperature,
            num_images_per_prompt=1
        ).images[0]
    
    return np.array(result)

def apply_upscaler(
    image: np.ndarray,
    prompt: str,
    temperature: float,
    selected_style: Optional[str] = None,
    output_size: Optional[Tuple[int, int]] = (1024, 1024)
) -> np.ndarray:
    pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(UPSCALER_MODEL, torch_dtype=dtype).to(device)
    
    input_image = Image.fromarray(image).resize((512, 512))
    
    with torch.no_grad():
        upscaled_image = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=20,
            guidance_scale=temperature
        ).images[0]
    
    upscaled_image = upscaled_image.resize(output_size, Image.LANCZOS)
    return np.array(upscaled_image)

def apply_controlnet(
    image: np.ndarray,
    prompt: str,
    temperature: float,
    selected_style: Optional[str] = None
) -> np.ndarray:
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
    
    control_image = get_canny_image(image)
    
    with torch.no_grad():
        output = pipe(
            prompt,
            image=control_image,
            num_inference_steps=20,
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
            control_guidance_start=CONTROL_GUIDANCE_START,
            control_guidance_end=CONTROL_GUIDANCE_END,
            guidance_scale=temperature
        ).images[0]
    
    return np.array(output)

def apply_pixart(
    image: np.ndarray,
    prompt: str,
    temperature: float,
    selected_style: Optional[str] = None,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    # For this example, we'll use the Freestyle model for Pixart as well
    pipe = DiffusionPipeline.from_pretrained(FREESTYLE_MODEL, torch_dtype=dtype).to(device)
    
    input_image = Image.fromarray(image).resize((1024, 1024))
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=20,
            guidance_scale=temperature,
            num_images_per_prompt=1
        ).images[0]
    
    return np.array(result)

def get_canny_image(image: np.ndarray) -> Image.Image:
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def load_style_prompts() -> List[Dict[str, str]]:
    try:
        with open(FREESTYLE_PROMPT_JSON, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading style prompts: {str(e)}")
        return []

def combine_prompts(user_prompt: str, style: Dict[str, str], max_tokens: int = 77) -> Tuple[str, str]:
    full_prompt = f"{user_prompt}, {style['prompt']}"
    neg_prompt = style.get('negative_prompt', '')
    
    # Simple truncation, you might want to implement more sophisticated tokenization
    if len(full_prompt) > max_tokens:
        full_prompt = full_prompt[:max_tokens]
    if len(neg_prompt) > max_tokens:
        neg_prompt = neg_prompt[:max_tokens]
    
    return full_prompt, neg_prompt