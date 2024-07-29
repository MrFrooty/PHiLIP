# model_downloader.py

import logging
from typing import List, Tuple
from diffusers import PixArtAlphaPipeline, StableDiffusionUpscalePipeline, ControlNetModel, DiffusionPipeline
from config import MODEL_MID_RES, MODEL_HIGH_RES, UPSCALER_MODEL, CONTROLNET_MODEL, LOG_FORMAT, LOG_DATE_FORMAT, FREESTYLE_MODEL

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

ModelInfo = Tuple[type, str]

def get_model_list() -> List[ModelInfo]:
    """
    Define the list of models to be downloaded.
    
    Returns:
        List[ModelInfo]: List of tuples containing model class and model name.
    """
    return [
        (PixArtAlphaPipeline, MODEL_MID_RES),
        (PixArtAlphaPipeline, MODEL_HIGH_RES),
        (StableDiffusionUpscalePipeline, UPSCALER_MODEL),
        (ControlNetModel, CONTROLNET_MODEL),
        (DiffusionPipeline, FREESTYLE_MODEL)
    ]

def download_model(model_class: type, model_name: str) -> None:
    """
    Download a single model.
    
    Args:
        model_class: The class of the model to be downloaded.
        model_name: The name or path of the model to be downloaded.
    """
    logger.info(f"Checking/downloading {model_name}")
    try:
        if model_class == DiffusionPipeline and model_name == FREESTYLE_MODEL:
            model_class.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        else:
            model_class.from_pretrained(model_name)
        logger.info(f"Model {model_name} is ready")
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")

def download_models() -> None:
    """Download and cache the required models."""
    models = get_model_list()
    for model_class, model_name in models:
        download_model(model_class, model_name)

if __name__ == "__main__":
    logger.info("Starting model download process...")
    try:
        download_models()
        logger.info("Model download process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the model download process: {str(e)}", exc_info=True)
    finally:
        logger.info("Model download process ended.")