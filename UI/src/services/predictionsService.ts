import { API_URL } from '../config';

export interface Prediction {
  average_probability: number;
  peak_probability: number;
  channel_probabilities: Record<string, number>;
}

export interface Predictions {
  week: Record<string, Prediction>;
  month: Record<string, Prediction>;
  quarter: Record<string, Prediction>;
}

export const fetchPredictions = async (): Promise<Predictions> => {
  try {
    const response = await fetch(`${API_URL}/predictions`);
    if (!response.ok) {
      throw new Error(`Failed to fetch predictions: ${response.statusText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }
}; 