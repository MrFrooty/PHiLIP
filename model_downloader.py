import logging
from typing import List, Tuple, Dict
from diffusers import PixArtAlphaPipeline, StableDiffusionUpscalePipeline, ControlNetModel, DiffusionPipeline
import os
from config import MODEL_MID_RES, MODEL_HIGH_RES, UPSCALER_MODEL, CONTROLNET_MODEL, FREESTYLE_MODEL

logger = logging.getLogger(__name__)

ModelInfo = Tuple[type, str]

def get_model_list() -> List[ModelInfo]:
    return [
        (PixArtAlphaPipeline, MODEL_MID_RES),
        (PixArtAlphaPipeline, MODEL_HIGH_RES),
        (StableDiffusionUpscalePipeline, UPSCALER_MODEL),
        (ControlNetModel, CONTROLNET_MODEL),
        (DiffusionPipeline, FREESTYLE_MODEL)
    ]

def download_model(model_class: type, model_name: str) -> None:
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
    models = get_model_list()
    for model_class, model_name in models:
        download_model(model_class, model_name)

def get_model_status() -> Dict[str, str]:
    status = {}
    models = get_model_list()
    for _, model_name in models:
        model_path = os.path.join(os.getcwd(), model_name)
        if os.path.exists(model_path):
            status[model_name] = "Downloaded"
        else:
            status[model_name] = "Not Downloaded"
    return status

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting model download process...")
    try:
        download_models()
        logger.info("Model download process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the model download process: {str(e)}", exc_info=True)
    finally:
        logger.info("Model download process ended.")