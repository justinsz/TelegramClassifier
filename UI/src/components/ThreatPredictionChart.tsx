import React, { FC, useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  ButtonGroup,
  Button,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useFilterContext, Filters } from '../contexts/FilterContext';
import { fetchPredictions, Predictions, Prediction } from '../services/predictionsService';
import { format, addDays, addWeeks, addMonths } from 'date-fns';

const ChartContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  height: '100%',
}));

const TimeRangeButton = styled(Button)(({ theme }) => ({
  '&.active': {
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
  },
}));

const TooltipContainer = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(1),
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
}));

interface DataPoint {
  date: string;
  [key: string]: string | number;
}

interface ThreatData {
  average_probability: number;
  peak_probability: number;
  channel_probabilities: number[];
}

interface TimeRangeData {
  [threatType: string]: ThreatData;
}

interface PredictionsData {
  week: TimeRangeData;
  month: TimeRangeData;
  quarter: TimeRangeData;
}

const THREAT_TYPE_MAPPING: Record<string, string> = {
  ddos: 'DDoS',
  ransomware_with_data_theft: 'Ransomware',
  data_exfiltration: 'Data Exfiltration',
  wiper: 'Wiper',
  fraud: 'Fraud',
  defacement: 'Defacement',
  cve: 'CVE'
};

const THREAT_COLORS: Record<string, string> = {
  ddos: '#FF6B6B',
  ransomware_with_data_theft: '#4ECDC4',
  data_exfiltration: '#45B7D1',
  wiper: '#96CEB4',
  fraud: '#FFEEAD',
  defacement: '#D4A5A5',
  cve: '#9B59B6'
};

const TIME_POINTS: Record<string, number> = {
  week: 7,    // 7 days
  month: 4,   // 4 weeks
  quarter: 3  // 3 months
};

const TIME_RANGE_DAYS: Record<string, number> = {
  week: 7,
  month: 30,
  quarter: 90
};

const convertPredictionsToData = (predictions: Predictions): PredictionsData => {
  const convertTimeRange = (timeRange: Record<string, Prediction>): TimeRangeData => {
    const result: TimeRangeData = {};
    for (const [threatType, prediction] of Object.entries(timeRange)) {
      result[threatType] = {
        average_probability: prediction.average_probability,
        peak_probability: prediction.peak_probability,
        channel_probabilities: Object.values(prediction.channel_probabilities)
      };
    }
    return result;
  };

  return {
    week: convertTimeRange(predictions.week),
    month: convertTimeRange(predictions.month),
    quarter: convertTimeRange(predictions.quarter)
  };
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  return (
    <TooltipContainer>
      <Typography variant="body2">{label}</Typography>
      {payload.map((entry: any, index: number) => (
        <Typography key={index} variant="body2" sx={{ color: entry.color }}>
          {`${entry.name}: ${entry.value.toFixed(1)}%`}
        </Typography>
      ))}
    </TooltipContainer>
  );
};

const ThreatPredictionChart: FC = () => {
  const [timeRange, setTimeRange] = useState<'week' | 'month' | 'quarter'>('week');
  const [data, setData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { filters } = useFilterContext();

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const predictions = await fetchPredictions();
        const processedPredictions = convertPredictionsToData(predictions);
        const timeRangeData = processedPredictions[timeRange];
        if (!timeRangeData) {
          throw new Error(`No data available for ${timeRange} time range`);
        }

        const numDays = TIME_RANGE_DAYS[timeRange];
        const processedData: DataPoint[] = Array.from({ length: numDays }, (_, i) => {
          const date = new Date();
          date.setDate(date.getDate() + i + 1);

          const dataPoint: DataPoint = {
            date: format(date, 'MM/dd/yyyy')
          };

          Object.entries(timeRangeData).forEach(([threatType, threatData]) => {
            if (filters[threatType as keyof Filters]) {
              const probability = threatData.channel_probabilities[i] || 0;
              dataPoint[THREAT_TYPE_MAPPING[threatType]] = probability * 100;
            }
          });

          return dataPoint;
        });

        setData(processedData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load prediction data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [timeRange, filters]);

  if (loading) {
    return <Typography>Loading predictions...</Typography>;
  }

  if (error) {
    return <Typography color="error">{error}</Typography>;
  }

  if (data.length === 0) {
    return <Typography>No prediction data available</Typography>;
  }

  return (
    <Box>
      <ButtonGroup variant="contained" sx={{ mb: 2 }}>
        <Button
          onClick={() => setTimeRange('week')}
          variant={timeRange === 'week' ? 'contained' : 'outlined'}
        >
          7 Days
        </Button>
        <Button
          onClick={() => setTimeRange('month')}
          variant={timeRange === 'month' ? 'contained' : 'outlined'}
        >
          1 Month
        </Button>
        <Button
          onClick={() => setTimeRange('quarter')}
          variant={timeRange === 'quarter' ? 'contained' : 'outlined'}
        >
          Quarter
        </Button>
      </ButtonGroup>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis 
            domain={[0, 100]} 
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {Object.entries(THREAT_TYPE_MAPPING).map(([key, label]) => (
            filters[key as keyof Filters] && (
              <Line
                key={key}
                type="monotone"
                dataKey={label}
                stroke={THREAT_COLORS[key]}
                dot={false}
              />
            )
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ThreatPredictionChart; 