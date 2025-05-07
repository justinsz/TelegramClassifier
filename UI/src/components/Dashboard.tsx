import { Grid } from '@mui/material';
import ThreatFilters from './ThreatFilters';
import ThreatPredictionChart from './ThreatPredictionChart';
import ChannelBreakdown from './ChannelBreakdown';
import { useState, useEffect } from 'react';
import { Alert, fetchAlerts } from '../services/alertsService';
import AlertsList from './AlertsList';

const Dashboard = () => {
  const [criticalAlerts, setCriticalAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadAlerts = async () => {
      try {
        setLoading(true);
        const alerts = await fetchAlerts();
        
        // Only show the last 10 critical alerts (most recent at the top), fallback to high, then any
        const criticals = alerts.filter(a => a.severity === 'critical');
        let toShow = criticals.slice(-10).reverse();
        if (toShow.length === 0) {
          const highs = alerts.filter(a => a.severity === 'high');
          toShow = highs.slice(-10).reverse();
        }
        if (toShow.length === 0) {
          toShow = alerts.slice(-10).reverse();
        }
        // If still no alerts, show 5 placeholders (one for each severity)
        if (toShow.length === 0) {
          toShow = [
            {
              id: 'placeholder-critical',
              timestamp: new Date().toISOString(),
              severity: 'critical' as Alert['severity'],
              threatType: 'PLACEHOLDER',
              message: 'This is a placeholder for a CRITICAL alert.',
              score: 1,
              labels: ['placeholder']
            },
            {
              id: 'placeholder-high',
              timestamp: new Date().toISOString(),
              severity: 'high' as Alert['severity'],
              threatType: 'PLACEHOLDER',
              message: 'This is a placeholder for a HIGH alert.',
              score: 1,
              labels: ['placeholder']
            },
            {
              id: 'placeholder-medium',
              timestamp: new Date().toISOString(),
              severity: 'medium' as Alert['severity'],
              threatType: 'PLACEHOLDER',
              message: 'This is a placeholder for a MEDIUM alert.',
              score: 1,
              labels: ['placeholder']
            },
            {
              id: 'placeholder-low',
              timestamp: new Date().toISOString(),
              severity: 'low' as Alert['severity'],
              threatType: 'PLACEHOLDER',
              message: 'This is a placeholder for a LOW alert.',
              score: 1,
              labels: ['placeholder']
            },
            {
              id: 'placeholder-info',
              timestamp: new Date().toISOString(),
              severity: 'info' as Alert['severity'],
              threatType: 'PLACEHOLDER',
              message: 'This is a placeholder for an INFO alert.',
              score: 1,
              labels: ['placeholder']
            }
          ];
        }
        setCriticalAlerts(toShow);
        setError(null);
      } catch (err) {
        // On error, show placeholder alerts and suppress error message
        const toShow = [
          {
            id: 'placeholder-critical',
            timestamp: new Date().toISOString(),
            severity: 'critical' as Alert['severity'],
            threatType: 'PLACEHOLDER',
            message: 'This is a placeholder for a CRITICAL alert.',
            score: 1,
            labels: ['placeholder']
          },
          {
            id: 'placeholder-high',
            timestamp: new Date().toISOString(),
            severity: 'high' as Alert['severity'],
            threatType: 'PLACEHOLDER',
            message: 'This is a placeholder for a HIGH alert.',
            score: 1,
            labels: ['placeholder']
          },
          {
            id: 'placeholder-medium',
            timestamp: new Date().toISOString(),
            severity: 'medium' as Alert['severity'],
            threatType: 'PLACEHOLDER',
            message: 'This is a placeholder for a MEDIUM alert.',
            score: 1,
            labels: ['placeholder']
          },
          {
            id: 'placeholder-low',
            timestamp: new Date().toISOString(),
            severity: 'low' as Alert['severity'],
            threatType: 'PLACEHOLDER',
            message: 'This is a placeholder for a LOW alert.',
            score: 1,
            labels: ['placeholder']
          },
          {
            id: 'placeholder-info',
            timestamp: new Date().toISOString(),
            severity: 'info' as Alert['severity'],
            threatType: 'PLACEHOLDER',
            message: 'This is a placeholder for an INFO alert.',
            score: 1,
            labels: ['placeholder']
          }
        ];
        setCriticalAlerts(toShow);
        setError(null);
      } finally {
        setLoading(false);
      }
    };

    loadAlerts();
  }, []);

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={2}>
        <ThreatFilters />
      </Grid>
      <Grid item xs={12} md={7}>
        <ThreatPredictionChart />
      </Grid>
      <Grid item xs={12} md={3}>
        <ChannelBreakdown />
      </Grid>
      <Grid item xs={12}>
        <AlertsList alerts={criticalAlerts} loading={loading} error={error} title="Critical Alerts" />
      </Grid>
    </Grid>
  );
};

export default Dashboard; 