import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from io import BytesIO
import base64
import os
import json
from datetime import datetime
from typing import List, Dict
import uuid

from image_generation_loop import generate_images
from image_enhancement import apply_enhancement
from config import (
    INITIAL_PROMPT, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE, MAX_LOG_SIZE, BACKUP_COUNT,
    ENHANCEMENT_OPTIONS, DEFAULT_TEMPERATURE, RESOLUTIONS, NUM_IMAGES_LIST, INFERENCE_STEPS_LIST,
    IMAGE_FOLDER
)
from model_downloader import download_models, get_model_status
from utils import encode_image, decode_image, handle_error
from progress_tracker import progress_tracker

app = Flask(__name__)
@app.before_request
def log_request_info():
    logger.info(f"Received request: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    logger.info(f"Body: {request.get_data()}")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def setup_logging():
    handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

@app.route('/generate', methods=['POST'])
def generate_images_endpoint():
    try:
        data = request.json
        prompt = data.get('prompt', INITIAL_PROMPT)
        num_images = data.get('numImages', 9)
        resolution = data.get('resolution', 512)
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)
        inference_steps = data.get('inferenceSteps', 50)
            
        images = generate_images(prompt, num_images, resolution, temperature, None, inference_steps)
        encoded_images = [f"data:image/png;base64,{encode_image(img)}" for img in images]
        return jsonify({'images': encoded_images})
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return jsonify(handle_error(e)), 500

def encode_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/enhance', methods=['POST'])
def enhance_image_endpoint():
    try:
        data = request.json
        image_data = data.get('imageData')
        prompt = data.get('prompt')
        enhancement_option = data.get('enhancementOption')
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)

        image = decode_image(image_data)
        enhanced_image = apply_enhancement(image, prompt, enhancement_option, temperature)
        encoded_image = encode_image(enhanced_image)
        return jsonify({'images': [f"data:image/png;base64,{encoded_image}"]})
    except Exception as e:
        logger.error(f"Error during image enhancement: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/settings', methods=['GET', 'PUT'])
def handle_settings():
    if request.method == 'GET':
        return jsonify({
            'initialPrompt': INITIAL_PROMPT,
            'defaultTemperature': DEFAULT_TEMPERATURE,
            'resolutions': RESOLUTIONS,
            'numImagesList': NUM_IMAGES_LIST,
            'inferenceStepsList': INFERENCE_STEPS_LIST,
            'enhancementOptions': ENHANCEMENT_OPTIONS
        })
    elif request.method == 'PUT':
        try:
            # Update settings logic here (if needed)
            return jsonify({'message': 'Settings updated successfully'})
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify(handle_error(e)), 500

@app.route('/enhancement-options', methods=['GET'])
def get_enhancement_options():
    return jsonify({'options': ENHANCEMENT_OPTIONS})

@app.route('/style-prompts', methods=['GET'])
def get_style_prompts():
    try:
        with open('style_prompt.json', 'r') as f:
            style_prompts = json.load(f)
        return jsonify(style_prompts)
    except Exception as e:
        logger.error(f"Error fetching style prompts: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/apply-freestyle', methods=['POST'])
def apply_freestyle_endpoint():
    try:
        data = request.json
        image_data = data.get('imageData')
        prompt = data.get('prompt')
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)
        selected_style = data.get('selectedStyle')
        
        try:
            image = decode_image(image_data)
        except ValueError as ve:
            logger.error(f"Error decoding image: {str(ve)}")
            return jsonify({"error": "Invalid image data. Please check the image and try again."}), 400
        
        enhanced_image = apply_enhancement(image, prompt, "Freestyle", temperature, selected_style)
        encoded_image = encode_image(enhanced_image)
        return jsonify({'enhancedImage': [f"data:image/png;base64,{encoded_image}"]})
    except Exception as e:
        logger.error(f"Error during Freestyle enhancement: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/apply-upscaler', methods=['POST'])
def apply_upscaler_endpoint():
    try:
        data = request.json
        image_data = data.get('imageData')
        prompt = data.get('prompt')
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)
        output_size = tuple(data.get('outputSize', (1024, 1024)))
        
        try:
            image = decode_image(image_data)
        except ValueError as ve:
            logger.error(f"Error decoding image: {str(ve)}")
            return jsonify({"error": "Invalid image data. Please check the image and try again."}), 400
        
        enhanced_image = apply_enhancement(image, prompt, "Upscaler", temperature, output_size=output_size)
        encoded_image = encode_image(enhanced_image)
        return jsonify({'enhancedImage': [f"data:image/png;base64,{encoded_image}"]})
    except Exception as e:
        logger.error(f"Error during Upscaler enhancement: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/apply-controlnet', methods=['POST'])
def apply_controlnet_endpoint():
    try:
        data = request.json
        image_data = data.get('imageData')
        prompt = data.get('prompt')
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)
        
        try:
            image = decode_image(image_data)
        except ValueError as ve:
            logger.error(f"Error decoding image: {str(ve)}")
            return jsonify({"error": "Invalid image data. Please check the image and try again."}), 400
        
        enhanced_image = apply_enhancement(image, prompt, "ControlNet", temperature)
        encoded_image = encode_image(enhanced_image)
        return jsonify({'enhancedImage': [f"data:image/png;base64,{encoded_image}"]})
    except Exception as e:
        logger.error(f"Error during ControlNet enhancement: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/apply-pixart', methods=['POST'])
def apply_pixart_endpoint():
    try:
        data = request.json
        image_data = data.get('imageData')
        prompt = data.get('prompt')
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)

        try:
            image = decode_image(image_data)
        except ValueError as ve:
            logger.error(f"Error decoding image: {str(ve)}")
            return jsonify({"error": "Invalid image data. Please check the image and try again."}), 400
        
        enhanced_image = apply_enhancement(image, prompt, "Pixart", temperature)
        encoded_image = encode_image(enhanced_image)
        # Log the response
        logger.info(f"Applied Pixart enhancement. Output image size: {enhanced_image.shape}")
        logger.info(f"Encoded image data URI length: {len(encoded_image)}")
        return jsonify({'enhancedImage': [f"data:image/png;base64,{encoded_image}"]})
    except Exception as e:
        logger.error(f"Error during Pixart enhancement: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/download-models', methods=['POST'])
def download_models_endpoint():
    try:
        download_models()
        return jsonify({'message': 'Models downloaded successfully'})
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/model-status', methods=['GET'])
def get_model_status_endpoint():
    try:
        status = get_model_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error fetching model status: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/save-image', methods=['POST'])
def save_image_endpoint():
    try:
        data = request.json
        image_data = data.get('imageData')
        file_name = data.get('fileName')

        if not file_name:
            file_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        image = decode_image(image_data)
        image_path = os.path.join(IMAGE_FOLDER, file_name)
        Image.fromarray(image).save(image_path)

        return jsonify({'message': 'Image saved successfully', 'path': image_path})
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/generated-images', methods=['GET'])
def get_generated_images():
    try:
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return jsonify({'images': image_files})
    except Exception as e:
        logger.error(f"Error fetching generated images: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/validate-input', methods=['POST'])
def validate_user_input():
    try:
        data = request.json
        input_value = data.get('input')
        input_type = data.get('inputType')
        options = data.get('options', {})

        # Implement input validation logic here
        # For now, we'll just return the input as valid
        is_valid = True
        message = "Input is valid"

        return jsonify({'isValid': is_valid, 'message': message, 'value': input_value})
    except Exception as e:
        logger.error(f"Error validating user input: {str(e)}")
        return jsonify(handle_error(e)), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        log_level = request.args.get('logLevel')

        # Implement log fetching logic here
        # For this example, we'll just read the last 100 lines of the log file
        with open(LOG_FILE, 'r') as f:
            logs = f.readlines()[-100:]

        return jsonify({'logs': logs})
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        return jsonify(handle_error(e)), 500
    
@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    progress = progress_tracker.get_progress(task_id)
    return jsonify({'progress': progress})
    
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Test endpoint is working"}), 200

if __name__ == "__main__":
    logger.info("Starting image generation server...")
    app.run(debug=True, host='0.0.0.0', port=7000)