# user_input_handler.py

import logging
from typing import Any, List, Union, Optional
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
            "Options: 'regenerate', 'reselect', 'stop', 'continue', 'prompt', 'temperature', 'restart', or 'change_num_images': "
        ).strip().lower()
        if user_input in VALID_USER_COMMANDS:
            return user_input
        else:
            logger.warning(f"Invalid option: {user_input}. Please enter a valid command.")

def get_user_input(
    prompt: str, 
    input_type: type, 
    min_value: Optional[Union[int, float]] = None, 
    max_value: Optional[Union[int, float]] = None, 
    valid_options: Optional[List[str]] = None
) -> Any:
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
                return validate_string_input(user_input, valid_options)
            
            value = input_type(user_input)
            
            if not is_value_in_range(value, min_value, max_value):
                raise ValueError(f"Input must be between {min_value} and {max_value}")
            
            return value
        except ValueError as e:
            logger.warning(f"Invalid input: {e}")

def validate_string_input(user_input: str, valid_options: List[str]) -> str:
    """
    Validate string input against a list of valid options.
    
    Args:
        user_input: The user's input string.
        valid_options: A list of valid options.
    
    Returns:
        str: The validated input string.
    
    Raises:
        ValueError: If the input is not in the list of valid options.
    """
    user_input_lower = user_input.lower()
    valid_options_lower = [option.lower() for option in valid_options]
    if user_input_lower in valid_options_lower:
        return valid_options[valid_options_lower.index(user_input_lower)]
    else:
        raise ValueError(f"Input must be one of: {', '.join(valid_options)}")

def is_value_in_range(value: Union[int, float], min_value: Optional[Union[int, float]], max_value: Optional[Union[int, float]]) -> bool:
    """
    Check if a value is within the specified range.
    
    Args:
        value: The value to check.
        min_value: The minimum allowed value.
        max_value: The maximum allowed value.
    
    Returns:
        bool: True if the value is within the range, False otherwise.
    """
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True