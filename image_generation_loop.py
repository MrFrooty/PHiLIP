# image_generation_loop.py

import os
import shutil
import logging
from typing import List, Optional
import numpy as np
from textwrap import dedent
from generate_image import generate_images
from display_image import display_and_select_image
from user_input_handler import handle_user_input
from config import (
    IMAGE_FOLDER, RESOLUTIONS, NUM_IMAGES_LIST, 
    INFERENCE_STEPS_LIST, DEFAULT_TEMPERATURE,
    LOG_FORMAT, LOG_DATE_FORMAT
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
    temperatures: List[float] = []
    selected_images: List[np.ndarray] = []
    generated_image_sets: List[List[np.ndarray]] = []
    
    for iteration in range(len(RESOLUTIONS)):
        resolution = RESOLUTIONS[iteration]
        num_images = NUM_IMAGES_LIST[iteration]
        inference_steps = INFERENCE_STEPS_LIST[iteration]
        
        base_images = selected_images if selected_images and iteration > 0 else None
        current_temperature = temperatures[iteration] if iteration < len(temperatures) else temperature

        logger.info(dedent(f"""
        Current settings:
        Prompt: {prompt}
        Temperature: {current_temperature}
        Resolution: {resolution}
        Inference steps: {inference_steps}
        """))

        generated_images = generate_images(prompt, num_images, resolution, current_temperature, base_images, inference_steps)
        generated_image_sets.append(generated_images)

        selected_image = display_and_select_image(generated_images, resolution, iteration)

        user_input = handle_user_input()

        if user_input == "regenerate":
            continue
        elif user_input == "restart":
            return []
        elif user_input == "reselect":
            iteration = max(0, iteration - 1)
            continue
        elif user_input == "stop":
            logger.info("User requested to stop. Exiting.")
            return None
        elif user_input == "prompt":
            prompt = input("Enter new prompt: ")
            continue
        elif user_input == "temperature":
            temperature = get_new_temperature()
            if iteration == len(temperatures):
                temperatures.append(temperature)
            else:
                temperatures[iteration] = temperature
            continue
        elif user_input == "continue":
            pass

        if selected_image is None:
            logger.warning("No image selected, exiting.")
            return None

        selected_images.extend(selected_image if isinstance(selected_image, list) else [selected_image])

        if iteration >= len(temperatures):
            temperatures.append(current_temperature)

    return selected_images

def get_new_temperature() -> float:
    """
    Get new temperature from user input.
    
    Returns:
        float: New temperature value.
    """
    while True:
        try:
            new_temp = float(input("Enter new temperature (suggested range from 0.5 to 1.5): "))
            return new_temp
        except ValueError:
            logger.warning("Invalid temperature input. Please enter a valid number.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    result = image_generation_loop("Test prompt")
    print(f"Final result: {result}")