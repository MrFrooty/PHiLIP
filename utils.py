import base64
import io
from PIL import Image
import numpy as np

def encode_image(image: np.ndarray) -> str:
    """
    Encode a numpy array image to a base64 string.
    
    Args:
        image (np.ndarray): The image as a numpy array.
    
    Returns:
        str: The base64 encoded string of the image.
    """
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_image(encoded_string: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array image.
    
    Args:
        encoded_string (str): The base64 encoded string of the image.
    
    Returns:
        np.ndarray: The decoded image as a numpy array.
    """
    decoded = base64.b64decode(encoded_string)
    img = Image.open(io.BytesIO(decoded))
    return np.array(img)

def handle_error(e: Exception) -> dict:
    """
    Handle exceptions and return a formatted error message.
    
    Args:
        e (Exception): The exception that was raised.
    
    Returns:
        dict: A dictionary containing the error message.
    """
    return {"error": str(e)}