import React from 'react';
import { AppBar, Toolbar, Typography, Box, Tabs, Tab } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';

const TopBar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleTabChange = (_: React.SyntheticEvent, newValue: string) => {
    navigate(newValue);
  };

  const currentPath = location.pathname;
  const value = currentPath === '/' ? '/dashboard' : currentPath;

  return (
    <AppBar position="fixed">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 0, mr: 4 }}>
          ThreatScope
        </Typography>
        
        <Box sx={{ flexGrow: 1 }}>
          <Tabs 
            value={value}
            onChange={handleTabChange}
            textColor="inherit"
            indicatorColor="secondary"
          >
            <Tab label="Dashboard" value="/dashboard" />
            <Tab label="Channels" value="/channels" />
            <Tab label="Alerts" value="/alerts" />
          </Tabs>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar; 