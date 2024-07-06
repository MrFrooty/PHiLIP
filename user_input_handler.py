# user_input_handler.py

import logging
from config import VALID_USER_INPUTS

logger = logging.getLogger(__name__)

def handle_user_input() -> str:
    """
    Handle user input for image generation process.
    
    Returns:
        str: Valid user input option.
    """
    while True:
        user_input = input(
            "Options: 'regenerate', 'reselect', 'stop', 'continue', 'prompt', 'temperature', or 'restart': "
        ).strip().lower()
        if user_input in VALID_USER_INPUTS:
            return user_input
        else:
            logger.warning(f"Invalid option: {user_input}. Please enter a valid command.")