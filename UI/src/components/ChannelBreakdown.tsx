import React, { useState } from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { styled } from '@mui/material/styles';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { useNavigate } from 'react-router-dom';

const BreakdownContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  height: '100%',
}));

interface DataItem {
  name: string;
  value: number;
  channelId?: string;
}

// Function to process channel data and group small values
const processChannelData = (data: { name: string; value: number; channelId?: string }[]): DataItem[] => {
  // Sort by value in descending order
  const sortedData = [...data].sort((a, b) => b.value - a.value);
  
  // Take top channels (those with >= 5% probability)
  const significantChannels = sortedData.filter(item => item.value >= 5);
  
  // Sum up the values of channels with < 5% probability
  const otherValue = sortedData
    .filter(item => item.value < 5)
    .reduce((sum, item) => sum + item.value, 0);
  
  // Return significant channels plus "Other" if there are small channels
  return [
    ...significantChannels,
    ...(otherValue > 0 ? [{ name: 'Other', value: Number(otherValue.toFixed(1)) }] : []),
  ];
};

const COLORS = [
  '#1976d2',
  '#2e7d32',
  '#ed6c02',
  '#9c27b0',
  '#d32f2f',
  '#0288d1',
  '#388e3c',
  '#f57c00',
  '#7b1fa2',
  '#c62828',
  '#0277bd',
  '#2e7d32',
  '#ef6c00',
  '#6a1b9a',
  '#b71c1c',
];

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    payload: DataItem;
  }>;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  return (
    <Box
      sx={{
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 1.5,
        border: '1px solid rgba(255, 255, 255, 0.2)',
        borderRadius: 1,
      }}
    >
      <Typography variant="body2" sx={{ color: '#fff' }}>
        {payload[0].name}
      </Typography>
      <Typography variant="body2" sx={{ color: '#90caf9' }}>
        {payload[0].value.toFixed(1)}%
      </Typography>
    </Box>
  );
};

const ChannelBreakdown: React.FC = () => {
  const navigate = useNavigate();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const handleMouseEnter = (index: number) => {
    setHoveredIndex(index);
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
  };

  // Sample data - replace with actual threat prediction data
  const data = processChannelData([
    { name: 'CVE Notify', value: 25.5, channelId: '13' },
    { name: 'Cyber Security News', value: 18.2, channelId: '1' },
    { name: 'The Hacker News', value: 15.7, channelId: '16' },
    { name: 'BleepingComputer', value: 12.3, channelId: '15' },
    { name: 'Malware News', value: 8.4, channelId: '14' },
    { name: 'CTI Now', value: 7.1, channelId: '17' },
    { name: 'InfoSec Experts', value: 4.8, channelId: '2' },
    { name: 'Cloud Security', value: 4.2, channelId: '3' },
    { name: 'Red Team Alerts', value: 3.8, channelId: '8' },
  ]);

  const handleChannelClick = (entry: DataItem) => {
    if (entry.channelId) {
      navigate('/channels');
    }
  };

  return (
    <BreakdownContainer>
      <Typography variant="h6" gutterBottom>
        Channel Breakdown
      </Typography>
      <Box sx={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={handleChannelClick}
              cursor="pointer"
            >
              {data.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                  opacity={hoveredIndex === null || hoveredIndex === index ? 1 : 0.5}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      </Box>
      <Box sx={{ mt: 2, maxHeight: 150, overflowY: 'auto' }}>
        {data.map((entry, index) => (
          <Box
            key={entry.name}
            sx={{
              display: 'flex',
              alignItems: 'center',
              mb: 1,
              cursor: entry.channelId ? 'pointer' : 'default',
              opacity: hoveredIndex === null || hoveredIndex === index ? 1 : 0.5,
              '&:hover': {
                opacity: 1,
              },
            }}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
            onClick={() => entry.channelId && handleChannelClick(entry)}
          >
            <Box
              sx={{
                width: 12,
                height: 12,
                backgroundColor: COLORS[index % COLORS.length],
                mr: 1,
                borderRadius: '50%',
              }}
            />
            <Typography
              variant="body2"
              sx={{
                flex: 1,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {entry.name}
            </Typography>
            <Typography variant="body2" sx={{ ml: 1, color: 'text.secondary' }}>
              {entry.value.toFixed(1)}%
            </Typography>
          </Box>
        ))}
      </Box>
    </BreakdownContainer>
  );
};

export default ChannelBreakdown;