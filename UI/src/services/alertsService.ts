import { API_URL } from '../config';

export interface Alert {
  id: string;
  timestamp: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  threatType: string;
  message: string;
  score: number;
  labels?: string[];
}

// Map criticality levels to severity
const criticalityToSeverity = {
  'CRITICAL': 'critical',
  'HIGH': 'high',
  'MEDIUM': 'medium',
  'LOW': 'low',
  'INFO': 'info'
} as const;

// Map threat types to more readable labels
const threatTypeLabels: { [key: string]: string } = {
  'ransomware_with_data_theft': 'Ransomware with Data Theft',
  'ransomware_no_data_theft': 'Ransomware',
  'ddos': 'DDoS Attack',
  'wiper': 'Wiper Malware',
  'data_exfiltration': 'Data Exfiltration',
  'fraud': 'Fraud',
  'defacement': 'Defacement',
  'cve': 'CVE'
};

export const processClassifiedMessage = (message: any): Alert => {
  // Generate a unique ID using timestamp and channel
  const id = `${message.timestamp}-${message.channel_username}`;
  
  // Convert labels to readable threat types
  const threatType = Array.isArray(message.labels) && message.labels.length > 0
    ? threatTypeLabels[message.labels[0]] || message.labels[0]
    : 'Unknown';

  return {
    id,
    timestamp: new Date(message.timestamp).getTime().toString(),
    severity: criticalityToSeverity[message.criticality as keyof typeof criticalityToSeverity] || 'medium',
    threatType,
    message: message.text,
    score: message.score || 0
  };
};

interface AlertFilters {
  severity?: string[];
  threatTypes?: string[];
  timeRange?: number;
  minScore?: number;
}

export const filterAlerts = (alerts: Alert[], filters: AlertFilters): Alert[] => {
  const now = Date.now();
  return alerts.filter(alert => {
    // Filter by severity
    if (filters.severity?.length && !filters.severity.includes(alert.severity)) {
      return false;
    }

    // Filter by threat type
    if (filters.threatTypes?.length && !filters.threatTypes.includes(alert.threatType)) {
      return false;
    }

    // Filter by time range
    if (filters.timeRange) {
      const hoursDiff = (now - new Date(alert.timestamp).getTime()) / (1000 * 60 * 60);
      if (hoursDiff > filters.timeRange) {
        return false;
      }
    }

    // Filter by minimum score
    if (filters.minScore !== undefined && alert.score < filters.minScore) {
      return false;
    }

    return true;
  });
};

export const sortAlerts = (alerts: Alert[], sortBy: 'time' | 'severity' | 'score' = 'time'): Alert[] => {
  const severityOrder: Record<Alert['severity'], number> = {
    critical: 4,
    high: 3,
    medium: 2,
    low: 1,
    info: 0,
  };

  return [...alerts].sort((a, b) => {
    switch (sortBy) {
      case 'severity':
        return severityOrder[b.severity] - severityOrder[a.severity];
      case 'score':
        return b.score - a.score;
      case 'time':
      default:
        return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    }
  });
};

export const fetchAlerts = async (): Promise<Alert[]> => {
  try {
    const response = await fetch(`${API_URL}/telegram_messages_labled.csv`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const text = await response.text();
    return parseCSV(text);
  } catch (error) {
    console.error('Error fetching alerts:', error);
    throw error;
  }
};

const parseCSV = (csv: string): Alert[] => {
  const lines = csv.split('\n');
  return lines.slice(1)
    .filter(line => line.trim())
    .map(line => {
      const values = line.split(',');
      if (values.length < 7 || !values[0] || !values[1] || !values[2]) {
        // Skip rows that don't have enough columns or missing id/timestamp/severity
        return null;
      }
      let labels: string[] = [];
      try {
        // Parse labels from the CSV (assuming it's in a format like "['label1', 'label2']")
        const labelsStr = values[6] || '[]';
        labels = JSON.parse(labelsStr.replace(/'/g, '"'));
      } catch (e) {
        console.warn('Failed to parse labels:', values[6]);
      }
      return {
        id: values[0],
        timestamp: values[1],
        severity: typeof values[2] === 'string' ? (values[2].toLowerCase() as Alert['severity']) : 'info',
        threatType: values[3] || 'Unknown',
        message: values[4] || '',
        score: values[5] ? parseFloat(values[5]) : 0,
        labels: Array.isArray(labels) ? labels : []
      } as Alert;
    })
    .filter((a): a is Alert => !!a && Array.isArray(a.labels));
}; 