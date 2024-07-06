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

# User input options
VALID_USER_INPUTS: Final[set] = {
    "regenerate", "reselect", "stop", "continue", 
    "prompt", "temperature", "restart"
}

# Initial prompt
INITIAL_PROMPT: Final[str] = """
A magical landscape with a towering, crystalline mountain and a serene lake 
reflecting iridescent clouds. At the lake's edge, a glowing blue-leaved tree 
stands amidst bioluminescent plants. A bridge of light arches over the lake 
towards an enchanted castle nestled in the mountain peaks.
"""

# Logging configuration
LOG_FORMAT: Final[str] = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: Final[str] = '%Y-%m-%d %H:%M:%S'