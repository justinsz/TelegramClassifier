import { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  styled,
} from '@mui/material';
import { Alert, fetchAlerts } from '../services/alertsService';
import { API_URL } from '../config';

const AlertsContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

interface AlertsListProps {
  alerts?: Alert[];
  loading?: boolean;
  error?: string | null;
  title?: string;
  standalone?: boolean;
}

const AlertsList = ({ alerts, loading, error, title = 'Recent Alerts', standalone = false }: AlertsListProps) => {
  const [localAlerts, setLocalAlerts] = useState<Alert[]>([]);
  const [localLoading, setLocalLoading] = useState(true);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    // If alerts are provided as props, use them
    if (alerts !== undefined) {
      setLocalAlerts(alerts);
      setLocalLoading(loading || false);
      setLocalError(error || null);
      return;
    }

    // Otherwise, if this is a standalone component, fetch its own data
    if (standalone) {
      const loadAlerts = async () => {
        try {
          setLocalLoading(true);
          const response = await fetch(`${API_URL}/telegram_messages_classified.csv`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const text = await response.text();
          const data = parseCSV(text);
          // Show the last 10 messages (most recent at the top)
          const last10 = data.slice(-10).reverse();
          setLocalAlerts(last10);
          setLocalError(null);
        } catch (err) {
          setLocalError('Failed to fetch data: ' + (err instanceof Error ? err.message : 'Unknown error'));
        } finally {
          setLocalLoading(false);
        }
      };

      loadAlerts();
    }
  }, [alerts, loading, error, standalone]);

  const parseCSV = (csv: string): Alert[] => {
    const lines = csv.split('\n');
    const parsed: (Alert | null)[] = lines.slice(1)
      .filter(line => line.trim())
      .map(line => {
        const values = line.split(',');
        if (values.length < 11 || !values[2]) {
          // Skip rows that don't have enough columns or missing id
          return null;
        }
        let labels: string[] = [];
        try {
          // Parse labels from the CSV (column indices based on the actual file structure)
          const labelsStr = values[7] || '[]';  // labels is in column 8 (index 7)
          labels = JSON.parse(labelsStr.replace(/'/g, '"'));
        } catch (e) {
          console.warn('Failed to parse labels:', values[7]);
        }
        // Extract a clean threat type
        let threatType = 'unknown';
        if (labels.length > 0) {
          const label = labels[0];
          if (typeof label === 'string' && label.includes(':')) {
            threatType = label.split(':')[1] || label.split(':')[0];
          } else if (typeof label === 'string') {
            threatType = label;
          }
        }
        // Robust date parsing
        let timestamp = values[3];
        let parsedDate = new Date(timestamp);
        if (!timestamp || isNaN(parsedDate.getTime())) {
          timestamp = 'Unknown Date';
        }
        // Robust severity assignment
        let severity: Alert['severity'] = 'info';
        if (values[10]) {
          const sev = values[10].toLowerCase();
          if (['high', 'low', 'critical', 'medium', 'info'].includes(sev)) {
            severity = sev as Alert['severity'];
          }
        }
        return {
          id: values[2] ? values[2] : 'unknown',  // message_id
          timestamp,      // robust timestamp
          severity,       // robust severity
          threatType,     // cleaned threat type
          message: values[4] ? values[4] : '',  // actual message text
          score: values[9] ? parseFloat(values[9]) : 0,  // usefulness_score
          labels
        };
      });
    return parsed.filter((a): a is Alert => a !== null);
  };

  if (localLoading) {
    return <Typography>Loading alerts...</Typography>;
  }

  if (localError) {
    return (
      <AlertsContainer>
        <Typography color="error">{localError}</Typography>
      </AlertsContainer>
    );
  }

  if (localAlerts.length === 0) {
    return (
      <AlertsContainer>
        <Typography>No alerts found.</Typography>
      </AlertsContainer>
    );
  }

  return (
    <Box>
      <AlertsContainer>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        {localAlerts.map((alert) => (
          <Box 
            key={alert.id} 
            sx={{ 
              mb: 2,
              p: 2,
              borderLeft: 6,
              borderColor: alert.severity === 'critical' ? 'error.main' : 'warning.main'
            }}
          >
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
              {alert.timestamp}
            </Typography>
            <Typography sx={{ color: alert.severity === 'critical' ? 'error.main' : 'warning.main', fontWeight: 'bold' }}>
              Severity: {alert.severity.toUpperCase()} | Type: {alert.threatType}
            </Typography>
            <Typography sx={{ mt: 1 }}>{alert.message}</Typography>
          </Box>
        ))}
      </AlertsContainer>
    </Box>
  );
};

export default AlertsList; 