import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
});

export const generateImages = async (prompt, numImages, resolution, temperature, inferenceSteps) => {
  try {
    const response = await api.post('/generate', { prompt, numImages, resolution, temperature, inferenceSteps });
    return response.data;
  } catch (error) {
    console.error('Error generating images:', error);
    throw error;
  }
};

export const enhanceImage = async (imageData, prompt, enhancementOption, temperature) => {
  try {
    const response = await api.post('/enhance', { imageData, prompt, enhancementOption, temperature });
    return response.data;
  } catch (error) {
    console.error('Error enhancing image:', error);
    throw error;
  }
};

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