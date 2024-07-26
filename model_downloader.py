# model_downloader.py

import logging
from diffusers import PixArtAlphaPipeline, StableDiffusionUpscalePipeline, ControlNetModel
from config import MODEL_MID_RES, MODEL_HIGH_RES, UPSCALER_MODEL, CONTROLNET_MODEL, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def download_models():
    """
    Download and cache the required models.
    """
    models = [
        (PixArtAlphaPipeline, MODEL_MID_RES),
        (PixArtAlphaPipeline, MODEL_HIGH_RES),
        (StableDiffusionUpscalePipeline, UPSCALER_MODEL),
        (ControlNetModel, CONTROLNET_MODEL)
    ]

    for model_class, model_name in models:
        logger.info(f"Checking/downloading {model_name}")
        try:
            model_class.from_pretrained(model_name)
            logger.info(f"Model {model_name} is ready")
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting model download process...")
    download_models()
    logger.info("Model download process completed.")