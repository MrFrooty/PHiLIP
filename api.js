import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
});

// Image Generation
export const generateImages = async (prompt, numImages, resolution, temperature, inferenceSteps) => {
  try {
    const response = await api.post('/generate', { prompt, numImages, resolution, temperature, inferenceSteps });
    return response.data;
  } catch (error) {
    console.error('Error generating images:', error);
    throw error;
  }
};

// Image Enhancement
export const enhanceImage = async (imageData, prompt, enhancementOption, temperature) => {
  try {
    const response = await api.post('/enhance', { imageData, prompt, enhancementOption, temperature });
    return response.data;
  } catch (error) {
    console.error('Error enhancing image:', error);
    throw error;
  }
};

// Settings Management
export const getGenerationSettings = async () => {
  try {
    const response = await api.get('/settings');
    return response.data;
  } catch (error) {
    console.error('Error fetching generation settings:', error);
    throw error;
  }
};

export const updateGenerationSettings = async (settings) => {
  try {
    const response = await api.put('/settings', settings);
    return response.data;
  } catch (error) {
    console.error('Error updating generation settings:', error);
    throw error;
  }
};

// Enhancement Options
export const getEnhancementOptions = async () => {
  try {
    const response = await api.get('/enhancement-options');
    return response.data;
  } catch (error) {
    console.error('Error fetching enhancement options:', error);
    throw error;
  }
};

// Style Prompts
export const getStylePrompts = async () => {
  try {
    const response = await api.get('/style-prompts');
    return response.data;
  } catch (error) {
    console.error('Error fetching style prompts:', error);
    throw error;
  }
};

// Freestyle Enhancement
export const applyFreestyle = async (imageData, prompt, selectedStyle, temperature) => {
  try {
    const response = await api.post('/apply-freestyle', { imageData, prompt, selectedStyle, temperature });
    return response.data;
  } catch (error) {
    console.error('Error applying Freestyle:', error);
    throw error;
  }
};

// Upscaler Enhancement
export const applyUpscaler = async (imageData, prompt, outputSize) => {
  try {
    const response = await api.post('/apply-upscaler', { imageData, prompt, outputSize });
    return response.data;
  } catch (error) {
    console.error('Error applying Upscaler:', error);
    throw error;
  }
};

// ControlNet Enhancement
export const applyControlNet = async (imageData, prompt) => {
  try {
    const response = await api.post('/apply-controlnet', { imageData, prompt });
    return response.data;
  } catch (error) {
    console.error('Error applying ControlNet:', error);
    throw error;
  }
};

// Pixart Enhancement
export const applyPixart = async (imageData, prompt, temperature) => {
  try {
    const response = await api.post('/apply-pixart', { imageData, prompt, temperature });
    return response.data;
  } catch (error) {
    console.error('Error applying Pixart:', error);
    throw error;
  }
};

// Model Management
export const downloadModels = async () => {
  try {
    const response = await api.post('/download-models');
    return response.data;
  } catch (error) {
    console.error('Error downloading models:', error);
    throw error;
  }
};

export const getModelStatus = async () => {
  try {
    const response = await api.get('/model-status');
    return response.data;
  } catch (error) {
    console.error('Error fetching model status:', error);
    throw error;
  }
};

// Image Handling
export const saveImage = async (imageData, fileName) => {
  try {
    const response = await api.post('/save-image', { imageData, fileName });
    return response.data;
  } catch (error) {
    console.error('Error saving image:', error);
    throw error;
  }
};

export const getGeneratedImages = async () => {
  try {
    const response = await api.get('/generated-images');
    return response.data;
  } catch (error) {
    console.error('Error fetching generated images:', error);
    throw error;
  }
};

// User Input Handling
export const validateUserInput = async (input, inputType, options) => {
  try {
    const response = await api.post('/validate-input', { input, inputType, options });
    return response.data;
  } catch (error) {
    console.error('Error validating user input:', error);
    throw error;
  }
};

// Logging
export const getLogs = async (startDate, endDate, logLevel) => {
  try {
    const response = await api.get('/logs', { params: { startDate, endDate, logLevel } });
    return response.data;
  } catch (error) {
    console.error('Error fetching logs:', error);
    throw error;
  }
};

export default {
  generateImages,
  enhanceImage,
  getGenerationSettings,
  updateGenerationSettings,
  getEnhancementOptions,
  getStylePrompts,
  applyFreestyle,
  applyUpscaler,
  applyControlNet,
  applyPixart,
  downloadModels,
  getModelStatus,
  saveImage,
  getGeneratedImages,
  validateUserInput,
  getLogs,
};