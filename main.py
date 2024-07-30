# main.py

import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
from image_generation_loop import image_generation_loop
from image_enhancement import apply_enhancement
from config import INITIAL_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE, MAX_LOG_SIZE, BACKUP_COUNT, ENHANCEMENT_OPTIONS, DEFAULT_TEMPERATURE, RESOLUTIONS, NUM_IMAGES_LIST, INFERENCE_STEPS_LIST   

app = Flask(__name__)
CORS(app)

def setup_logging() -> logging.Logger:
    """Set up logging with rotation."""
    handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

@app.route('/generate', methods=['POST'])
def generate_images():
    data = request.json
    prompt = data.get('prompt', INITIAL_PROMPT)
    num_images = data.get('numImages', 1)
    resolution = data.get('resolution', 512)
    temperature = data.get('temperature', 1.0)
    inference_steps = data.get('inferenceSteps', 50)

    try:
        images = image_generation_loop(prompt, num_images, resolution, temperature, inference_steps)
        return jsonify({'images': images})
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def enhance_image():
    data = request.json
    image_data = data.get('imageData')
    prompt = data.get('prompt')
    enhancement_option = data.get('enhancementOption')
    temperature = data.get('temperature', 1.0)

    try:
        enhanced_image = apply_enhancement(image_data, prompt, enhancement_option, temperature)
        return jsonify({'enhancedImage': enhanced_image})
    except Exception as e:
        logger.error(f"Error during image enhancement: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/settings', methods=['GET', 'PUT'])
def handle_settings():
    if request.method == 'GET':
        # Return current settings
        return jsonify({
            'initialPrompt': INITIAL_PROMPT,
            'defaultTemperature': DEFAULT_TEMPERATURE,
            'resolutions': RESOLUTIONS,
            'numImagesList': NUM_IMAGES_LIST,
            'inferenceStepsList': INFERENCE_STEPS_LIST,
            'enhancementOptions': ENHANCEMENT_OPTIONS
        })
    elif request.method == 'PUT':
        # Update settings (you might want to implement this in a separate configuration file)
        data = request.json
        # Update settings logic here
        return jsonify({'message': 'Settings updated successfully'})

if __name__ == "__main__":
    logger.info("Starting image generation server...")
    app.run(debug=True)