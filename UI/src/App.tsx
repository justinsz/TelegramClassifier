import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

import TopBar from './components/TopBar';
import Dashboard from './components/Dashboard';
import ChannelsView from './components/ChannelsView';
import AlertsList from './components/AlertsList';
import { FilterProvider } from './contexts/FilterContext';

// Create dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#1a1a1a',
      paper: '#2d2d2d',
    },
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#ce93d8',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const MainContainer = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  marginTop: theme.spacing(8),
  minHeight: 'calc(100vh - 64px)',
  backgroundColor: theme.palette.background.default,
}));

const App = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <FilterProvider>
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <TopBar />
            <MainContainer>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/channels" element={<ChannelsView />} />
                <Route path="/alerts" element={<AlertsList standalone title="High & Critical Alerts" />} />
              </Routes>
            </MainContainer>
          </Box>
        </FilterProvider>
      </Router>
    </ThemeProvider>
  );
};

export default App; 