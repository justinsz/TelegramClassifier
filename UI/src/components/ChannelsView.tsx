import React, { useState } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  Link,
  Rating,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import LaunchIcon from '@mui/icons-material/Launch';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { channels } from '../data/channels_data';

interface Channel {
  id: number;
  name: string;
  telegramUrl: string;
  credibilityScore: number;
  credibilityCategory: string;
  relevanceScore: number;
  relevanceCategory: string;
  engagementScore: number;
  engagementCategory: string;
  frequencyScore: number;
  frequencyCategory: string;
  subscriberScore: number;
  audienceCategory: string;
}

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  '&.MuiTableCell-head': {
    backgroundColor: theme.palette.background.paper,
    color: theme.palette.text.primary,
    fontWeight: 'bold',
  },
}));

const StyledRating = styled(Rating)({
  '& .MuiRating-iconFilled': {
    color: '#90caf9',
  },
});

const getCategoryColor = (category: string): string => {
  switch (category.toLowerCase()) {
    case 'very high':
      return '#2e7d32';
    case 'high':
      return '#1976d2';
    case 'moderate':
      return '#ed6c02';
    case 'medium':
      return '#ed6c02';
    case 'low':
      return '#d32f2f';
    case 'very large':
      return '#2e7d32';
    case 'large':
      return '#1976d2';
    case 'small':
      return '#ed6c02';
    case 'very small':
      return '#d32f2f';
    default:
      return '#757575';
  }
};

const ChannelsView: React.FC = () => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Transform the raw channel data into properly typed objects
  const channelData: Channel[] = channels.map(channel => ({
    id: Number(channel[0]),
    name: String(channel[1]),
    telegramUrl: String(channel[2]),
    credibilityScore: Number(channel[3]),
    credibilityCategory: String(channel[4]),
    relevanceScore: Number(channel[5]),
    relevanceCategory: String(channel[6]),
    engagementScore: Number(channel[7]),
    engagementCategory: String(channel[8]),
    frequencyScore: Number(channel[9]),
    frequencyCategory: String(channel[10]),
    subscriberScore: Number(channel[11]),
    audienceCategory: String(channel[12]),
  }));

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      <TableContainer sx={{ maxHeight: 'calc(100vh - 200px)' }}>
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              <StyledTableCell>Channel Name</StyledTableCell>
              <StyledTableCell align="center">Credibility</StyledTableCell>
              <StyledTableCell align="center">Relevance</StyledTableCell>
              <StyledTableCell align="center">Engagement</StyledTableCell>
              <StyledTableCell align="center">Frequency</StyledTableCell>
              <StyledTableCell align="center">Audience</StyledTableCell>
              <StyledTableCell align="center">Actions</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {channelData
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((channel) => (
                <TableRow key={channel.id} hover>
                  <TableCell component="th" scope="row">
                    {channel.name}
                  </TableCell>
                  <TableCell align="center">
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                      <StyledRating
                        value={Number(channel.credibilityScore) / 2}
                        precision={0.5}
                        readOnly
                        size="small"
                      />
                      <Chip
                        label={channel.credibilityCategory}
                        size="small"
                        sx={{
                          backgroundColor: getCategoryColor(channel.credibilityCategory),
                          color: 'white',
                        }}
                      />
                    </Box>
                  </TableCell>
                  <TableCell align="center">
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                      <StyledRating
                        value={Number(channel.relevanceScore) / 2}
                        precision={0.5}
                        readOnly
                        size="small"
                      />
                      <Chip
                        label={channel.relevanceCategory}
                        size="small"
                        sx={{
                          backgroundColor: getCategoryColor(channel.relevanceCategory),
                          color: 'white',
                        }}
                      />
                    </Box>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={channel.engagementCategory}
                      size="small"
                      sx={{
                        backgroundColor: getCategoryColor(channel.engagementCategory),
                        color: 'white',
                      }}
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={channel.frequencyCategory}
                      size="small"
                      sx={{
                        backgroundColor: getCategoryColor(channel.frequencyCategory),
                        color: 'white',
                      }}
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={channel.audienceCategory}
                      size="small"
                      sx={{
                        backgroundColor: getCategoryColor(channel.audienceCategory),
                        color: 'white',
                      }}
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                      <Tooltip title="View channel details">
                        <IconButton
                          size="small"
                          onClick={() => {/* TODO: Implement channel details view */}}
                        >
                          <InfoOutlinedIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Open in Telegram">
                        <IconButton
                          size="small"
                          component={Link}
                          href={channel.telegramUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <LaunchIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={channelData.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
};

export default ChannelsView;