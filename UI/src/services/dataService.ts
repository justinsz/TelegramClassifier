import Papa, { ParseResult } from 'papaparse';
import { processClassifiedMessage, Alert } from './alertsService';
import { API_URL } from '../config';

interface ClassifiedMessage {
  labels: string[] | string;
  matching_keywords: string[] | string;
  usefulness_score: number;
  criticality: string;
  timestamp: string;
  channel: string;
  channel_username: string;
  text: string;
  channel_id?: number;
}

export const fetchClassifiedMessages = async (): Promise<Alert[]> => {
  try {
    const response = await fetch(`${API_URL}/telegram_messages_classified.csv`);
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('No classified messages found. Please run the classifier first.');
      }
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }

    const csvText = await response.text();
    if (!csvText.trim()) {
      throw new Error('The CSV file is empty');
    }
    
    return new Promise((resolve, reject) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results: ParseResult<ClassifiedMessage>) => {
          if (results.errors.length > 0) {
            console.warn('CSV parsing warnings:', results.errors);
          }

          if (!results.data || results.data.length === 0) {
            reject(new Error('No data found in the CSV file'));
            return;
          }

          const alerts = results.data
            .filter((row) => {
              // Filter out invalid rows
              if (!row || !row.labels || !row.usefulness_score) {
                return false;
              }

              // Filter out rows without labels or with usefulness score below threshold
              return (
                row.labels && 
                row.labels.length > 0 && 
                row.usefulness_score >= 0.25 &&
                row.criticality !== 'INFO'
              );
            })
            .map((row) => {
              // Convert string array representation to actual array
              if (typeof row.labels === 'string') {
                try {
                  row.labels = JSON.parse(row.labels.replace(/'/g, '"'));
                } catch (e) {
                  console.warn('Failed to parse labels:', row.labels);
                  row.labels = [];
                }
              }
              
              if (typeof row.matching_keywords === 'string') {
                try {
                  row.matching_keywords = JSON.parse(row.matching_keywords.replace(/'/g, '"'));
                } catch (e) {
                  console.warn('Failed to parse matching_keywords:', row.matching_keywords);
                  row.matching_keywords = [];
                }
              }
              
              return processClassifiedMessage(row);
            });

          if (alerts.length === 0) {
            console.warn('No alerts passed the filtering criteria');
          } else {
            console.log(`Successfully loaded ${alerts.length} alerts`);
          }
            
          resolve(alerts);
        },
        error: (error: Error) => {
          reject(new Error(`Failed to parse CSV: ${error.message}`));
        }
      });
    });
  } catch (error) {
    console.error('Error fetching classified messages:', error);
    throw error;
  }
}; 