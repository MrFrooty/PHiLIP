# image_generation_loop.py

import os
import shutil
import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from textwrap import dedent
from generate_image import generate_images
from display_image import display_and_select_image, save_images
from user_input_handler import handle_user_input, get_user_input
from image_enhancement import apply_enhancement
from config import (
    IMAGE_FOLDER, RESOLUTIONS, NUM_IMAGES_LIST, 
    INFERENCE_STEPS_LIST, DEFAULT_TEMPERATURE,
    LOG_FORMAT, LOG_DATE_FORMAT, TEMPERATURE_PROMPT,
    INFERENCE_STEPS_PROMPT, NUM_IMAGES_PROMPT,
    ENHANCEMENT_PROMPT, ENHANCEMENT_OPTIONS
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def clear_generated_images_folder() -> None:
    """Clears all files in the generated_images folder."""
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)
    os.makedirs(IMAGE_FOLDER)

def image_generation_loop(initial_prompt: str) -> Optional[List[np.ndarray]]:
    """
    Main loop for image generation process.
    
    Args:
        initial_prompt: Initial prompt for image generation.
    
    Returns:
        Optional[List[np.ndarray]]: List of final selected images or None if process is stopped.
    """
    clear_generated_images_folder()

    prompt = initial_prompt
    temperature = DEFAULT_TEMPERATURE
    resolution = RESOLUTIONS[0]  # Start with 512x512
    num_images = NUM_IMAGES_LIST[0]
    inference_steps = INFERENCE_STEPS_LIST[0]
    enhanced_image = None
    enhancement_option = None
    base_images = None

    while True:
        if base_images is None:
            log_current_settings(prompt, temperature, resolution, inference_steps, num_images)
            base_images = generate_images(prompt, num_images, resolution, temperature, None, inference_steps)

        selected_images = display_and_select_image(base_images, resolution, 0)
        if not selected_images:
            logger.warning("No images selected. Exiting program.")
            return None

        base_image = selected_images[0]

        if enhancement_option is None:
            enhancement_option = get_user_input(ENHANCEMENT_PROMPT, str, valid_options=ENHANCEMENT_OPTIONS)
        
        enhanced_image = apply_enhancement(base_image, prompt, enhancement_option, temperature)

        final_resolution = 1024 if enhancement_option in ["Upscaler", "Pixart", "ControlNet"] else resolution
        save_images([enhanced_image], final_resolution, final=True)
        logger.info(f"Final enhanced image saved as final-enhanced-{final_resolution}.png")

        user_action = handle_user_input()
        if user_action == "stop":
            logger.info("User requested to stop. Exiting program.")
            return [enhanced_image]
        elif user_action == "regenerate":
            enhanced_image = regenerate_enhanced_image(base_image, prompt, enhancement_option, temperature, final_resolution)
        elif user_action == "restart":
            base_images, enhanced_image, enhancement_option = reset_generation_process()
        elif user_action == "reselect":
            logger.info("Reselecting base image...")
            continue
        elif user_action == "change_temp":
            temperature = get_user_input(TEMPERATURE_PROMPT, float, 0.5, 1.5)
            base_images = None
        elif user_action == "change_prompt":
            prompt = input("Enter new prompt: ")
            base_images = None
        elif user_action == "change_steps":
            inference_steps = get_user_input(INFERENCE_STEPS_PROMPT, int, 1, 100)
            base_images = None
        elif user_action == "change_num_images":
            num_images = get_user_input(NUM_IMAGES_PROMPT, int, 1, 9)
            base_images = None
        elif user_action == "continue":
            return [enhanced_image]

def log_current_settings(prompt: str, temperature: float, resolution: int, inference_steps: int, num_images: int) -> None:
    """Log current generation settings."""
    logger.info(dedent(f"""
    Current settings:
    Prompt: {prompt}
    Temperature: {temperature}
    Resolution: {resolution}
    Inference steps: {inference_steps}
    Number of images: {num_images}
    """))

def regenerate_enhanced_image(base_image: np.ndarray, prompt: str, enhancement_option: str, temperature: float, final_resolution: int) -> np.ndarray:
    """Regenerate the enhanced image."""
    logger.info("Regenerating enhanced image...")
    enhanced_image = apply_enhancement(base_image, prompt, enhancement_option, temperature)
    save_images([enhanced_image], final_resolution, final=True)
    logger.info(f"Regenerated enhanced image saved to {IMAGE_FOLDER}")
    return enhanced_image

def reset_generation_process() -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray], Optional[str]]:
    """Reset the generation process."""
    logger.info("Restarting the process...")
    return None, None, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    result = image_generation_loop("Test prompt")
    print(f"Final result: {result}")