# main.py

import logging
from image_generation_loop import image_generation_loop
from config import INITIAL_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main function to run the image generation process."""
    final_images = image_generation_loop(INITIAL_PROMPT)
    if final_images:
        logger.info("Image generation completed successfully.")
    else:
        logger.info("Program exiting.")

if __name__ == "__main__":
    main()