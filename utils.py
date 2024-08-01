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
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3] 
    
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_image(encoded_string: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array image.
    
    Args:
        encoded_string (str): The base64 encoded string of the image,
                              or a full data URI.
    
    Returns:
        np.ndarray: The decoded image as a numpy array.
    """
    if encoded_string.startswith('data:image'):
        encoded_string = encoded_string.split(',', 1)[1]
    
    try:
        decoded = base64.b64decode(encoded_string)
        img = Image.open(io.BytesIO(decoded))
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")

def handle_error(e: Exception) -> dict:
    """
    Handle exceptions and return a formatted error message.
    
    Args:
        e (Exception): The exception that was raised.
    
    Returns:
        dict: A dictionary containing the error message.
    """
    return {"error": str(e)}