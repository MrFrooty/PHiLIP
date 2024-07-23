# main.py

import logging
from logging.handlers import RotatingFileHandler
from image_generation_loop import image_generation_loop
from config import INITIAL_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE, MAX_LOG_SIZE, BACKUP_COUNT

def setup_logging():
    """Set up logging with rotation."""
    handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

def main() -> None:
    """Main function to run the image generation process."""
    logger = setup_logging()
    logger.info("Starting image generation process...")
    final_images = image_generation_loop(INITIAL_PROMPT)
    if final_images:
        logger.info("Image generation completed successfully.")
    else:
        logger.info("Program exiting.")

if __name__ == "__main__":
    main()