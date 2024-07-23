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
UPSCALER_MODEL: Final[str] = "stabilityai/stable-diffusion-x4-upscaler"

# User input options
VALID_USER_COMMANDS: Final[set] = {
    "regenerate", "reselect", "stop", "continue", 
    "prompt", "temperature", "restart"
}

# Initial prompt
INITIAL_PROMPT: Final[str] = """
A serene forest at dawn with towering, ancient trees draped in moss. A crystal-clear river winds through the landscape, reflecting the soft morning light. At the river's edge, vibrant flowers bloom amid bioluminescent mushrooms. A delicate stone bridge arches over the river, leading to a hidden grove with a mystical waterfall cascading into a shimmering pool.
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
ENHANCEMENT_PROMPT: Final[str] = "Select enhancement option (freestyle/pixart/upscale/none): "