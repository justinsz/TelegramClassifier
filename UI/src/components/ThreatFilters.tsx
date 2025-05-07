import React from 'react';
import { Paper, Typography, FormGroup, FormControlLabel, Checkbox } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useFilterContext } from '../contexts/FilterContext';
import type { Filters } from '../contexts/FilterContext';

const FilterContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
}));

const THREAT_TYPES = [
  { id: 'ddos', label: 'DDoS Attack' },
  { id: 'ransomware_with_data_theft', label: 'Ransomware with Data Theft' },
  { id: 'data_exfiltration', label: 'Data Exfiltration' },
  { id: 'wiper', label: 'Wiper Malware' },
  { id: 'fraud', label: 'Fraud' },
  { id: 'defacement', label: 'Defacement' },
  { id: 'cve', label: 'CVE' },
] as const;

type ThreatType = typeof THREAT_TYPES[number]['id'];

const ThreatFilters = () => {
  const { filters, setFilters } = useFilterContext();

  const handleThreatTypeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    setFilters((prev: Filters) => ({
      ...prev,
      [name as ThreatType]: checked
    }));
  };

  return (
    <FilterContainer>
      <Typography variant="h6" gutterBottom>
        Filter by
      </Typography>
      <FormGroup>
        {THREAT_TYPES.map(({ id, label }) => (
          <FormControlLabel
            key={id}
            control={
              <Checkbox 
                checked={filters[id as ThreatType]} 
                onChange={handleThreatTypeChange} 
                name={id}
              />
            }
            label={label}
          />
        ))}
      </FormGroup>
    </FilterContainer>
  );
};

export default ThreatFilters; 