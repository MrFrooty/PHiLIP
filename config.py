# config.py

import os
from typing import List, Final
import logging

# Image generation settings
RESOLUTIONS: Final[List[int]] = [512, 1024]
NUM_IMAGES_LIST: Final[List[int]] = [9, 1]
INFERENCE_STEPS_LIST: Final[List[int]] = [4, 10]
DEFAULT_TEMPERATURE: Final[float] = 1.0

# File paths
IMAGE_FOLDER: Final[str] = "generated_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Model names
MODEL_MID_RES: Final[str] = "PixArt-alpha/PixArt-XL-2-512x512"
MODEL_HIGH_RES: Final[str] = "PixArt-alpha/PixArt-XL-2-1024-MS"
UPSCALER_MODEL: Final[str] = "stabilityai/sd-x2-latent-upscaler"
SD_BASE_MODEL: Final[str] = "CompVis/stable-diffusion-v1-4"
FREESTYLE_MODEL: Final[str] = "stabilityai/stable-diffusion-xl-base-1.0"  

# ControlNet settings
CONTROLNET_MODEL: Final[str] = "lllyasviel/sd-controlnet-canny"
CONTROLNET_CONDITIONING_SCALE: Final[float] = 0.5
CONTROL_GUIDANCE_START: Final[float] = 0.0
CONTROL_GUIDANCE_END: Final[float] = 1.0

# Freestyle settings
FREESTYLE_PROMPT_JSON: Final[str] = "./style_prompt.json"
FREESTYLE_N: Final[int] = 160
FREESTYLE_B: Final[float] = 2.5
FREESTYLE_S: Final[int] = 1

# User input options
VALID_USER_COMMANDS: Final[set] = {
    "regenerate", "reselect", "stop", "continue", 
    "prompt", "temperature", "restart", "change_num_images"
}

# Enhancement options
ENHANCEMENT_OPTIONS: Final[List[str]] = ["Freestyle", "Upscaler", "ControlNet", "Pixart", "None"]

# Initial prompt
INITIAL_PROMPT: Final[str] = """
Explore the Shadow Dungeon, a dark cavern lit by spectral blue flames and lined with ancient runes. Navigate a labyrinth filled with shadow creatures to reach the Shadow Monarch's throne room, dominated by an obsidian throne and intense magical energy. 
"""

# Logging configuration
LOG_FORMAT: Final[str] = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: Final[str] = '%Y-%m-%d %H:%M:%S'
LOG_FILE: Final[str] = 'image_generation.log'
MAX_LOG_SIZE: Final[int] = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT: Final[int] = 3

# User input prompts
TEMPERATURE_PROMPT: Final[str] = "Enter new temperature (suggested range from 0.5 to 1.5): "
INFERENCE_STEPS_PROMPT: Final[str] = "Enter new number of inference steps (4-50 recommended): "
NUM_IMAGES_PROMPT: Final[str] = "Enter new number of images to generate (1-9 recommended): "
ENHANCEMENT_PROMPT: Final[str] = "Select enhancement option (Freestyle/Upscaler/ControlNet/Pixart/None): "