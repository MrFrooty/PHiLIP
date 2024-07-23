# user_input_handler.py

import logging
from typing import Any, List, Union
from config import VALID_USER_COMMANDS

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
        if user_input in VALID_USER_COMMANDS:
            return user_input
        else:
            logger.warning(f"Invalid option: {user_input}. Please enter a valid command.")

def get_user_input(prompt: str, input_type: type, min_value: Union[int, float] = None, max_value: Union[int, float] = None, valid_options: List[str] = None) -> Any:
    """
    Get and validate user input.
    
    Args:
        prompt: The prompt to display to the user.
        input_type: The type of input expected (int, float, or str).
        min_value: The minimum allowed value for numeric inputs.
        max_value: The maximum allowed value for numeric inputs.
        valid_options: A list of valid options for string inputs.
    
    Returns:
        The validated user input.
    """
    while True:
        user_input = input(prompt).strip()
        try:
            if input_type == str and valid_options:
                if user_input.lower() in valid_options:
                    return user_input.lower()
                else:
                    raise ValueError(f"Input must be one of: {', '.join(valid_options)}")
            
            value = input_type(user_input)
            
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                raise ValueError(f"Input must be between {min_value} and {max_value}")
            
            return value
        except ValueError as e:
            logger.warning(f"Invalid input: {e}")