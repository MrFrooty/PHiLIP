# PixArt Image Generation Project

## Overview

This project utilizes PixArt to generate images based on text prompts. It provides an interactive interface for users to generate, select, and refine images through multiple iterations.

## Table of Contents

1. [Setup](#setup)
2. [Running the Project](#running-the-project)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)
8. [License](#license)

## Setup

1. Create a new conda environment: conda create --name pixart python=3.8
2. Activate the conda environment: conda activate pixart
3. Install PyTorch and related packages: conda install -c pytorch pytorch torchvision torchaudio
4. Install other required packages: pip install -U diffusers transformers accelerate safetensors sentencepiece beautifulsoup4 matplotlib ftfy pillow requests scipy tqdm

## Running the Project

After setting up the environment and installing all required packages, run the project using: python main.py

This will start the image generation process using the initial prompt defined in `config.py`.

## Project Structure

- `main.py`: The entry point of the application.
- `config.py`: Contains configuration settings for the project.
- `image_generation_loop.py`: Implements the main image generation loop.
- `generate_image.py`: Handles the actual image generation using PixArt.
- `display_image.py`: Manages image display and user selection.
- `user_input_handler.py`: Handles user input during the generation process.

## Usage

1. The program will start by generating a set of images based on the initial prompt.
2. Images will be displayed, and you'll be prompted to select your favorites.
3. You can then choose to:
   - Regenerate images
   - Reselect from the previous set
   - Modify the prompt
   - Adjust the temperature
   - Continue to the next iteration
   - Restart the process
   - Stop the program

4. The process continues until you decide to stop or all iterations are completed.
5. Selected images are saved in the `generated_images` folder.

## Configuration

You can modify various settings in the `config.py` file:

- `RESOLUTIONS`: List of image resolutions for each iteration
- `NUM_IMAGES_LIST`: Number of images to generate in each iteration
- `INFERENCE_STEPS_LIST`: Number of inference steps for each iteration
- `DEFAULT_TEMPERATURE`: Default temperature for image generation
- `INITIAL_PROMPT`: The starting prompt for image generation

## Troubleshooting

- Ensure you have sufficient GPU memory if you're using CUDA for faster processing.
- If you encounter CUDA out of memory errors, try reducing the number of images generated or the resolution.
- For any persistent issues, check the error logs and consult the PixArt and Diffusers documentation.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any additional questions or support, please open an issue in the GitHub repository.