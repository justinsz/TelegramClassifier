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
- Generates detailed Excel reports with formatting

## Threat Categories

The system currently classifies messages into the following categories:
- DDoS Attacks
- Ransomware (with and without data theft)
- Wiper Malware
- Data Exfiltration
- Fraud
- Defacement

## Scoring System

Messages are scored based on three main dimensions, with scores rounded to 4 decimal places (e.g., 0.1324). Only messages with threat labels receive a score.

### 1. Threat Relevance (50%)
- **Base Score for Labeled Messages (25%)**: Automatic score for any message with threat labels
- **Multiple Threat Types Bonus (15%)**: Additional 7.5% per unique threat type (e.g., ransomware + data theft)
- **Keyword Density (10%)**: Measures the presence of relevant threat keywords
- **Critical Keywords Bonus**: Additional 10% if message contains critical indicators like:
  - Zero-day vulnerabilities
  - Active exploitation
  - Critical severity
  - Mass attacks
  - Emergency patches
  - Immediate action required
- **Link Bonus**: Additional 15% if message contains a URL (http:// or https://)
- **Guide/Tutorial Penalty**: Messages containing "guide", "tutorial", or "how to" are automatically classified as INFO

### 2. Engagement Score (30%)
- **Forwards (20%)**: Logarithmic scale based on number of forwards
  - 1-10 forwards: 0.20 - 0.40
  - 11-25 forwards: 0.41 - 0.60
  - 26-50 forwards: 0.61 - 0.80
  - 50+ forwards: 0.81 - 1.00
- **Replies (10%)**: Logarithmic scale based on number of replies
  - 1-5 replies: 0.10 - 0.30
  - 6-15 replies: 0.31 - 0.60
  - 16-25 replies: 0.61 - 0.80
  - 25+ replies: 0.81 - 1.00

### 3. Context Quality (20%)
- **Message Completeness (10%)**: Checks for key elements:
  - Who (attacker, group, organization)
  - What (vulnerability, exploit, attack)
  - When (discovery, detection time)
  - Where (affected systems, networks)
  - Bonus 5% for having all elements
- **Source Credibility (10%)**: Based on channel reputation
  - Top-tier sources (e.g., MalwareResearch): 1.0
  - High-tier sources (e.g., TheHackerNews): 0.9
  - Standard sources: 0.6

### Additional Scoring Factors

- **High Impact Bonus**: Messages with both high threat relevance (≥0.4) and high engagement (≥0.3) receive an additional 10% bonus
- **Score Capping**: All individual scores and the final score are capped at 1.0

### Criticality Classification

Based on the usefulness score, messages are classified into the following criticality levels:

| Criticality | Score Range | Description |
|-------------|-------------|-------------|
| CRITICAL    | 0.6000 - 1.0000 | Immediate attention required, high-impact threats |
| HIGH        | 0.4000 - 0.5999 | Significant threats requiring prompt action |
| MEDIUM      | 0.3000 - 0.3999 | Important threats that should be monitored |
| LOW         | 0.2000 - 0.2999 | Minor threats or informational messages |
| INFO        | 0.0000 - 0.1999 | General information or low-priority updates |

The Excel output includes color-coding for criticality levels:
- CRITICAL: Red
- HIGH: Light Red
- MEDIUM: Yellow
- LOW: Light Green
- INFO: Blue

### Output Features

The script generates three sets of output files, each containing the following columns:
- Channel name
- Channel username
- Message ID
- Timestamp
- Message text
- Number of forwards
- Reply count
- Threat labels
- **Matched keywords** (new): Lists all keywords that triggered the classification
- Usefulness score
- Criticality level

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

## Output

The script generates three sets of output files:

1. **Complete Dataset CSV** (`telegram_messages_classified.csv`):
   - A simple comma-separated values file containing all messages
   - Suitable for quick data analysis and processing
   - Contains both labeled and unlabeled messages

2. **Complete Dataset Excel** (`telegram_messages_classified.xlsx`):
   - A formatted Excel workbook with all messages
   - Enhanced features:
     - Color-coded threat categories
     - Conditional formatting for usefulness scores
     - Auto-filtering capabilities
     - Frozen headers for easy navigation

3. **Labeled Messages Only** (Ranked by Usefulness):
   - CSV format (`telegram_messages_labeled.csv`):
     - Contains only messages with identified threat labels
     - Sorted by usefulness score in descending order
     - Perfect for focusing on relevant threats
   
   - Excel format (`telegram_messages_labeled.xlsx`):
     - Same data as the labeled CSV but with enhanced formatting
     - Includes all Excel features of the complete dataset
     - Prioritized view of threats by usefulness

All files contain the following columns:
- Channel name
- Channel username
- Message ID
- Timestamp
- Message text
- Number of forwards
- Reply count
- Threat labels
- Matched keywords
- Usefulness score

## Security Note

Never commit your `.env` file or expose your Telegram API credentials. The `.gitignore` file is configured to prevent accidental commits of sensitive information.

## License

MIT License

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