# image_generation_loop.py

import os
import shutil
import logging
from typing import List, Optional
import numpy as np
from PIL import Image
from textwrap import dedent
from generate_image import generate_images
from display_image import display_and_select_image, save_images
from user_input_handler import handle_user_input, get_user_input
from image_enhancement import upscale_image, apply_freestyle  
from config import (
    IMAGE_FOLDER, RESOLUTIONS, NUM_IMAGES_LIST, 
    INFERENCE_STEPS_LIST, DEFAULT_TEMPERATURE,
    LOG_FORMAT, LOG_DATE_FORMAT, TEMPERATURE_PROMPT,
    INFERENCE_STEPS_PROMPT, NUM_IMAGES_PROMPT,
    ENHANCEMENT_PROMPT
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

    while True:
        logger.info(dedent(f"""
        Current settings:
        Prompt: {prompt}
        Temperature: {temperature}
        Resolution: {resolution}
        Inference steps: {inference_steps}
        Number of images: {num_images}
        """))

        generated_images = generate_images(prompt, num_images, resolution, temperature, None, inference_steps)

        user_action = get_user_action()
        if user_action == "stop":
            logger.info("User requested to stop. Exiting program.")
            return None
        elif user_action == "regenerate":
            logger.info("Regenerating images...")
            clear_generated_images_folder()
            continue
        elif user_action == "change_temp":
            temperature = get_user_input(TEMPERATURE_PROMPT, float, 0.5, 1.5)
            continue
        elif user_action == "change_prompt":
            prompt = input("Enter new prompt: ")
            continue
        elif user_action == "change_steps":
            inference_steps = get_user_input(INFERENCE_STEPS_PROMPT, int, 1, 100)
            continue
        elif user_action == "change_num_images":
            num_images = get_user_input(NUM_IMAGES_PROMPT, int, 1, 9)
            continue
        elif user_action == "continue":
            selected_images = display_and_select_image(generated_images, resolution, 0)
            if not selected_images:
                logger.warning("No images selected. Exiting program.")
                return None
            break

    enhancement_option = get_user_input(ENHANCEMENT_PROMPT, str, valid_options=["freestyle", "pixart", "upscale", "none"])
    if enhancement_option == "freestyle":
        selected_images = [apply_freestyle(selected_images[0], prompt)]
        upscale_option = input("Do you want to upscale or use PixArt 1024? (upscale/pixart/none): ").strip().lower()
        if upscale_option == "upscale":
            selected_images = [upscale_image(Image.fromarray(selected_images[0]), prompt)]
        elif upscale_option == "pixart":
            selected_images = generate_images(prompt, 1, 1024, temperature, selected_images, 10)
    elif enhancement_option == "pixart":
        selected_images = generate_images(prompt, 1, 1024, temperature, selected_images, 10)
    elif enhancement_option == "upscale":
        selected_images = [upscale_image(Image.fromarray(selected_images[0]), prompt)]
    
    # Save the final enhanced image
    if selected_images and enhancement_option != "none":
        final_resolution = 1024 if enhancement_option in ["pixart", "upscale"] else 512
        save_images(selected_images, final_resolution, final=True)
        logger.info(f"Final enhanced image saved to {IMAGE_FOLDER}")

    return selected_images

def get_user_action() -> str:
    """
    Handle user input after image generation.
    
    Returns:
        str: User's chosen action.
    """
    actions = {
        "1": "stop",
        "2": "regenerate",
        "3": "continue",
        "4": "change_temp",
        "5": "change_prompt",
        "6": "change_steps",
        "7": "change_num_images"
    }
    
    while True:
        print("\nAvailable actions:")
        for key, value in actions.items():
            print(f"{key}. {value}")
        
        action = input("Choose an action: ").strip().lower()
        if action in actions.values():
            return action
        elif action in actions:
            return actions[action]
        else:
            logger.warning(f"Invalid action: {action}. Please enter a valid option.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    result = image_generation_loop("Test prompt")
    print(f"Final result: {result}")