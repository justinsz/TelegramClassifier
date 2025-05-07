const express = require('express');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 3001;

// Enable CORS for development
app.use(cors());

// Add error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: err.message });
});

// Serve static files from the React app
app.use(express.static(path.join(__dirname, 'dist')));

// Serve the CSV file
app.get('/telegram_messages_classified.csv', (req, res) => {
  const csvPath = path.join(__dirname, '..', 'telegram_messages_classified.csv');
  console.log('Attempting to serve CSV file from:', csvPath);
  
  try {
    // Check if file exists
    if (!fs.existsSync(csvPath)) {
      console.error('CSV file not found at:', csvPath);
      return res.status(404).json({
        error: 'CSV file not found. Please run the classifier first.',
        path: csvPath
      });
    }

    // Get file stats
    const stats = fs.statSync(csvPath);
    console.log('CSV file stats:', {
      size: stats.size,
      modified: stats.mtime
    });

    // Set headers
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Length', stats.size);

    // Stream the file
    const fileStream = fs.createReadStream(csvPath);
    fileStream.on('error', (err) => {
      console.error('Error streaming file:', err);
      res.status(500).json({ error: 'Error reading CSV file', details: err.message });
    });

    fileStream.pipe(res);
  } catch (err) {
    console.error('Error handling CSV request:', err);
    res.status(500).json({ error: 'Server error', details: err.message });
  }
});

// Serve the labled CSV file
app.get('/telegram_messages_labled.csv', (req, res) => {
  const csvPath = path.join(__dirname, '..', 'telegram_messages_labled.csv');
  console.log('Attempting to serve labled CSV file from:', csvPath);

  try {
    if (!fs.existsSync(csvPath)) {
      console.error('Labled CSV file not found at:', csvPath);
      return res.status(404).json({
        error: 'Labled CSV file not found.',
        path: csvPath
      });
    }

    const stats = fs.statSync(csvPath);
    console.log('Labled CSV file stats:', {
      size: stats.size,
      modified: stats.mtime
    });

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Length', stats.size);

    const fileStream = fs.createReadStream(csvPath);
    fileStream.on('error', (err) => {
      console.error('Error streaming labled file:', err);
      res.status(500).json({ error: 'Error reading labled CSV file', details: err.message });
    });

    fileStream.pipe(res);
  } catch (err) {
    console.error('Error handling labled CSV request:', err);
    res.status(500).json({ error: 'Server error', details: err.message });
  }
});

// Serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(port, () => {
  const csvPath = path.join(__dirname, '..', 'telegram_messages_classified.csv');
  console.log(`Server is running on port ${port}`);
  console.log('Looking for CSV file at:', csvPath);
  if (fs.existsSync(csvPath)) {
    const stats = fs.statSync(csvPath);
    console.log('CSV file found!', {
      size: stats.size,
      modified: stats.mtime
    });
  } else {
    console.log('Warning: CSV file not found. Please run the classifier first.');
  }
}); 