import axios from 'axios';

// Remove any trailing slash from the API URL to prevent double slashes
const API_URL = process.env.REACT_APP_API_URL ? process.env.REACT_APP_API_URL.replace(/\/+$/, '') : 'https://1cf5-34-168-113-29.ngrok-free.app'; 

export const legalAssistantApi = {
  // First phase: Generate questions based on initial query
  generateQuestions: async (query) => {
    try {
      const response = await axios.post(`${API_URL}/generate_part1`, { query });
      return response.data;
    } catch (error) {
      console.error('Error generating questions:', error);
      throw error;
    }
  },

  // Second phase: Generate legal analysis based on query, case summary, and answers
  generateAnalysis: async (query, case_summary, answers) => {
    try {
      const response = await axios.post(`${API_URL}/generate_part2`, {
        query,
        case_summary,
        answers
      });
      return response.data;
    } catch (error) {
      console.error('Error generating analysis:', error);
      throw error;
    }
  },

  // For testing: Get full analysis in one call
  getFullAnalysis: async (query) => {
    try {
      const response = await axios.post(`${API_URL}/full_analysis`, { query });
      return response.data;
    } catch (error) {
      console.error('Error getting full analysis:', error);
      throw error;
    }
  }
};
