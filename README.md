# Telegram Message Classifier

This project scrapes messages from Telegram channels and classifies them based on cybersecurity threats and their severity.

## Prerequisites

- Python 3.7+
- Telegram API credentials (api_id and api_hash)
- Telegram account

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## First-Time Setup

1. Create a `.env` file in the project root with your Telegram API credentials:
```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=your_phone_number
```

2. Run the main script:
```bash
python3 main.py
```

3. First-time authentication:
   - The script will ask for your phone number
   - You'll need to enter your Telegram password
   - A one-time code will be sent to your Telegram account
   - Enter the code when prompted
   
   Note: This authentication process only happens on first run. Subsequent runs will use the saved session.

## Features

- Scrapes messages from multiple Telegram channels
- Classifies messages into threat categories
- Calculates usefulness scores based on multiple factors
- Generates detailed CSV reports
- Automatically filters out informational content (guides, tutorials, lists, etc.)
- Modular keyword management through separate data file

## Threat Categories

The system currently classifies messages into the following categories:
- DDoS Attacks
- Ransomware (with and without data theft)
- Wiper Malware
- Data Exfiltration
- Fraud
- Defacement

## Scoring System

Messages are scored based on two main dimensions, with scores rounded to 4 decimal places (e.g., 0.1324).

### 1. Threat Relevance (70%)
- **Base Score**: 0.35 for labeled messages, 0.60 for messages with critical indicators
- **Multiple Threat Types Bonus**: Up to 30% additional score
- **Keyword Density**: Up to 25% based on matching keywords
- **Critical Keywords Bonus**: 25% for critical threat indicators
- **Link Bonus**: 25% if message contains a URL
- **Technical Vulnerability Scores**:
  - SQL Injection: 0.75
  - Remote Code Execution: 0.80
  - Zero-day: 0.85
  - Other critical vulnerabilities: 0.75
- **Severity Multipliers**:
  - Ransomware with data theft: 1.0
  - Wiper: 1.0
  - Ransomware without data theft: 0.9
  - Data exfiltration: 0.9
  - DDoS: 0.8
  - Fraud: 0.7
  - Defacement: 0.6

### 2. Engagement Score (20%)
- **Forwards (70% weight)**:
  - Logarithmic scale with threshold at 25 forwards
  - Maximum score of 0.5
- **Replies (30% weight)**:
  - Logarithmic scale with threshold at 15 replies
  - Maximum score of 0.3

### Additional Scoring Features
- **Combined Bonus**: 10% additional score for messages with both high threat relevance (≥0.5) and high engagement (≥0.3)
- **Critical Threat Minimum**: Messages with critical patterns or indicators are guaranteed a minimum score of 0.55

### Criticality Levels
- **CRITICAL**: ≥ 0.55
- **HIGH**: ≥ 0.35
- **MEDIUM**: ≥ 0.25
- **LOW**: ≥ 0.15
- **INFO**: < 0.15 or informational content

## Keyword Management

Keywords and patterns are managed in `keywords_data.py`, which includes:
- Threat category keywords
- Technical vulnerability patterns
- Critical threat indicators
- Informational content filters
- User-generated content patterns
- Severity weights and thresholds

## Output Files

The system generates the following output files:

1. `telegram_messages_classified.csv`: Contains all messages with their classifications and scores, sorted by timestamp (newest first)
2. `telegram_messages_labeled.csv`: Contains only messages that have been classified as threats, sorted by usefulness score (highest first)
3. `telegram_messages_labeled.xlsx`: Excel version of labeled messages
4. `telegram_messages_classified.xlsx`: Excel version of all messages

## Informational Content Filtering

The system automatically filters out:
- Guides and tutorials
- Educational content
- Best practices and tips
- User-generated content
- Step-by-step instructions
- General informational resources

## Requirements

- Python 3.7+
- Required packages:
  - telethon
  - pandas
  - python-dotenv
  - xlsxwriter

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Telegram API credentials:
```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=your_phone_number
```

3. Add your target channels to `channels_data.py`

## Usage

1. Run the scraper:
```bash
python main.py
```

2. Classify messages:
```bash
python classify_messages.py
```

## Output Format

All output files contain the following columns:
- Channel name
- Channel username
- Message ID
- Timestamp
- Message text
- Forwards count
- Reply count
- Labels (threat categories)
- Matching keywords
- Usefulness score (formatted as 0.xxxx)
- Criticality level

## Security Note

Never commit your `.env` file or expose your Telegram API credentials. The `.gitignore` file is configured to prevent accidental commits of sensitive information.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This means you are free to:

- Share: Copy and redistribute the material in any medium or format
- Adapt: Remix, transform, and build upon the material

Under the following terms:

- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- NonCommercial: You may not use the material for commercial purposes

For more details, see the [LICENSE](LICENSE) file.

## Acknowledgments

- [Telethon](https://github.com/LonamiWebs/Telethon) for Telegram API access
- [Pandas](https://pandas.pydata.org/) for data processing
- [XlsxWriter](https://xlsxwriter.readthedocs.io/) for Excel file generation

## Configuration

You can modify the following in the code:
- `max_messages` in `main.py`: Number of messages to scrape per channel
- `channels` in `channels_data.py`: List of channels to monitor
- Classification parameters in `classify_messages.py`

## Contributing

Feel free to submit issues and enhancement requests.

# ThreatScope Dashboard

## Features

- **Backend** serves both `telegram_messages_classified.csv` and `telegram_messages_labled.csv` via dedicated routes (`/telegram_messages_classified.csv` and `/telegram_messages_labled.csv`).
- **Frontend Dashboard**:
  - Always displays up to 10 critical alerts (most recent at the top).
  - If there are no critical alerts, falls back to high alerts, then any alerts.
  - If no data is available, displays 5 clearly marked placeholder alerts (one for each severity: CRITICAL, HIGH, MEDIUM, LOW, INFO).
  - Channel Breakdown and Threat Prediction Chart are robust to missing or malformed data.
- **CORS-enabled**: The frontend and backend can be accessed from other devices on the same network.
- **Robust CSV Handling**: The dashboard is resilient to missing or malformed CSV data and will always show a meaningful UI.

## Usage

1. Place your `telegram_messages_classified.csv` and/or `telegram_messages_labled.csv` in the project root (same level as `UI`).
2. Start the backend and frontend:
   - From the `UI` directory, run:
     ```sh
     npm start
     ```
   - This will start both the backend and frontend servers.
3. Access the dashboard from your browser at `http://<host-ip>:3000` (replace `<host-ip>` with your machine's IP address).

## Troubleshooting

- **404 Error for CSV**: If you see `Failed to fetch data: HTTP error! status: 404`, make sure the relevant CSV file exists in the project root and is named exactly as expected (`telegram_messages_classified.csv` or `telegram_messages_labled.csv`).
- **Placeholder Alerts**: If no real alert data is available, the dashboard will display 5 placeholder alerts (one for each severity). This ensures the UI is always populated and makes it clear when no real data is present.
- **Network Access**: Ensure your firewall allows access to ports 3000 (frontend) and 3001 (backend) for other devices on your network.

## Updating Data

- To update the dashboard with new data, simply replace the CSV files in the project root and refresh the dashboard in your browser.
- The backend will automatically serve the latest CSV data.

## Notes

- **Placeholder Alerts**: These are shown when no real data is available and are clearly marked as `PLACEHOLDER` in the alert list. They help users and developers verify that the dashboard UI is working even when data is missing.
- **CSV Format**: Ensure your CSV files match the expected format. If you change the column order or add new fields, update the parsing logic in `alertsService.ts` and `AlertsList.tsx` accordingly.

---

For further customization or troubleshooting, see the code comments or contact the project maintainer.