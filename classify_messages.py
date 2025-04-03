import pandas as pd
import numpy as np
from datetime import datetime
import math
import re

# Keyword lists with expanded cybersecurity-specific terms
label_keywords = {
    "ddos": [
        "ddos", "denial of service", "botnet", "amplification", "udp flood", "http flood", "takedown",
        "volumetric attack", "traffic flood", "service disruption", "network flood", "ddos attack",
        "distributed denial", "traffic amplification", "service outage"
    ],
    "ransomware_with_data_theft": [
        "ransomware", "double extortion", "data leak", "leaked", "published victims", "exfiltrated",
        "ransom demand", "encrypted files", "data stolen", "victim data", "ransomware gang",
        "ransomware group", "data published", "leak site", "threatened to leak"
    ],
    "ransomware_no_data_theft": [
        "ransomware", "encryptor", "decryptor", "locker", "payload", "encrypted files",
        "encryption key", "file encryption", "crypto locker", "system locked", "ransom note",
        "bitcoin demand", "crypto demand"
    ],
    "wiper": [
        "wiper", "overwrite", "data destroy", "killdisk", "destroyed permanently",
        "wiping malware", "destructive malware", "data wiped", "disk wiper", "master boot record",
        "mbr wiper", "permanent deletion", "irreversible damage"
    ],
    "data_exfiltration": [
        "exfiltration", "stolen data", "siphoned", "dumped credentials", "extracted data",
        "data breach", "information theft", "data stolen", "credential leak", "sensitive data",
        "confidential data", "data compromise", "unauthorized access", "data exposed"
    ],
    "fraud": [
        "scam", "phishing", "smishing", "vishing", "impersonation", "fraudulent",
        "social engineering", "fake login", "credential theft", "spoofed", "fake domain",
        "malicious email", "fraud attempt", "business email compromise", "bec attack"
    ],
    "defacement": [
        "defaced", "defacement", "hacked homepage", "vandalized", "website graffiti",
        "site defacement", "web defacement", "page altered", "website vandalism",
        "compromised website", "modified homepage", "website takeover"
    ]
}

# Add these constants at the top of the file
THREAT_SEVERITY_WEIGHTS = {
    "ransomware_with_data_theft": 1.0,
    "wiper": 1.0,
    "ransomware_no_data_theft": 0.9,
    "data_exfiltration": 0.9,
    "ddos": 0.8,
    "fraud": 0.7,
    "defacement": 0.6
}

# Update the criticality thresholds
CRITICALITY_THRESHOLDS = {
    "CRITICAL": 0.55,
    "HIGH": 0.35,
    "MEDIUM": 0.25,
    "LOW": 0.15,
    "INFO": 0.0
}

# Technical vulnerabilities that should always be classified as critical
TECHNICAL_VULNERABILITIES = {
    "sql injection": 0.75,
    "sqli": 0.75,
    "remote code execution": 0.80,
    "rce": 0.80,
    "privilege escalation": 0.75,
    "authentication bypass": 0.75,
    "command injection": 0.75,
    "buffer overflow": 0.75,
    "memory corruption": 0.75,
    "arbitrary code execution": 0.80,
    "zero-day": 0.85,
    "0day": 0.85,
    "zero day": 0.85
}

# Critical keywords that indicate high-severity threats
critical_keywords = [
    # Zero-day and critical vulnerabilities
    "zero-day", "0day", "zero day", "critical vulnerability", "critical vulnerabilities",
    "high-severity vulnerability", "severe vulnerability", "critical security",
    
    # Active exploitation
    "active exploit", "actively exploited", "in the wild", "mass attack",
    "widespread attack", "under attack", "ongoing attack", "active campaign",
    
    # Technical threats
    "remote code execution", "rce vulnerability", "code injection", "sql injection",
    "command injection", "privilege escalation", "authentication bypass",
    "arbitrary code execution", "buffer overflow", "memory corruption",
    
    # Impact indicators
    "critical", "severe", "emergency", "urgent", "high-risk", "immediate action",
    "patch immediately", "update immediately", "exploited", "compromised",
    
    # Data breaches and exfiltration
    "breach", "leak", "stolen", "exposed", "unauthorized access", "data theft",
    "sensitive data exposed", "credentials leaked", "customer data",
    
    # Malware and ransomware
    "ransomware", "wiper", "malware", "backdoor", "rootkit", "botnet",
    "cryptominer", "data theft", "exfiltration", "encrypted files"
]

# Technical threat patterns that should always be considered critical
CRITICAL_PATTERNS = [
    r"CVE-\d{4}-\d{4,7}.*critical",
    r"remote code execution|RCE",
    r"SQL injection|SQLi",
    r"privilege escalation",
    r"zero[- ]day|0day",
    r"arbitrary code execution",
    r"authentication bypass",
    r"command injection",
    r"buffer overflow",
    r"memory corruption"
]

# Key elements to look for in messages
KEY_ELEMENTS = {
    "who": ["attacker", "group", "actor", "organization", "company"],
    "what": ["vulnerability", "exploit", "attack", "breach", "compromise"],
    "when": ["discovered", "detected", "reported", "occurred"],
    "where": ["system", "network", "application", "service", "infrastructure"]
}

def classify_message(message: str):
    """Classify a message into one or more categories based on keywords."""
    if not isinstance(message, str):
        return []
    
    message = message.lower()
    labels = []

    if any(k in message for k in label_keywords["ddos"]):
        labels.append("ddos")

    if any(k in message for k in label_keywords["ransomware_with_data_theft"]):
        labels.append("ransomware_with_data_theft")
    elif any(k in message for k in label_keywords["ransomware_no_data_theft"]):
        labels.append("ransomware_no_data_theft")

    if any(k in message for k in label_keywords["wiper"]):
        labels.append("wiper")

    if any(k in message for k in label_keywords["data_exfiltration"]):
        labels.append("data_exfiltration")

    if any(k in message for k in label_keywords["fraud"]):
        labels.append("fraud")

    if any(k in message for k in label_keywords["defacement"]):
        labels.append("defacement")

    return labels

def get_matching_keywords(message):
    """Extract all matching keywords from the message."""
    if not isinstance(message, str):
        return []
    
    message = message.lower()
    matches = []
    
    # Check each category's keywords
    for category, keywords in label_keywords.items():
        for keyword in keywords:
            if keyword in message:
                matches.append(f"{category}:{keyword}")
    
    # Check critical keywords
    for keyword in critical_keywords:
        if keyword in message:
            matches.append(f"critical:{keyword}")
    
    return matches

def calculate_threat_relevance(row):
    """Calculate threat relevance score (50%)."""
    if not isinstance(row['text'], str):
        return 0.0
        
    message = row['text'].lower()
    
    # Check for guide/tutorial content first (automatic INFO classification)
    if "guide" in message or "tutorial" in message or "how to" in message:
        return 0.0
    
    # Check for technical vulnerabilities first
    base_score = 0.0
    for vuln, score in TECHNICAL_VULNERABILITIES.items():
        if vuln in message:
            base_score = max(base_score, score)  # Use the highest score if multiple vulnerabilities are found
    
    if base_score > 0:
        # For technical vulnerabilities, we still want to consider other factors
        # but ensure the score stays above the CRITICAL threshold
        
        # Multiple threat types bonus (15%)
        threat_diversity = 0.0
        if isinstance(row['labels'], list):
            unique_threats = len(row['labels'])
            if unique_threats > 1:
                threat_diversity = min(0.15, 0.075 * unique_threats)
        
        # Keyword density bonus (10%)
        total_keywords = sum(len(keywords) for keywords in label_keywords.values())
        found_keywords = sum(1 for keywords in label_keywords.values() for kw in keywords if kw in message)
        keyword_density = (found_keywords / total_keywords if total_keywords > 0 else 0) * 0.1
        
        # Link bonus (5%)
        link_bonus = 0.05 if "http://" in message or "https://" in message else 0.0
        
        # Calculate final score while maintaining minimum technical vulnerability score
        return min(1.0, max(base_score, base_score + threat_diversity + keyword_density + link_bonus))
    
    # If no technical vulnerabilities found, proceed with regular scoring
    critical_indicators = [
        "critical vulnerability", "active exploit", "actively exploited",
        "in the wild", "mass attack", "widespread attack", "under attack",
        "ransomware attack", "data breach", "massive breach",
        "critical severity", "emergency patch"
    ]
    
    # Technical patterns check
    has_technical_pattern = any(re.search(pattern, message, re.IGNORECASE) for pattern in CRITICAL_PATTERNS)
    
    # If message contains critical indicators or technical patterns, ensure high base score
    if has_technical_pattern or any(indicator in message for indicator in critical_indicators):
        base_score = 0.6
    else:
        base_score = 0.35 if isinstance(row['labels'], list) and len(row['labels']) > 0 else 0.0
    
    # Multiple threat types bonus (30%)
    threat_diversity = 0.0
    if isinstance(row['labels'], list):
        unique_threats = len(row['labels'])
        if unique_threats > 1:
            threat_diversity = min(0.3, 0.15 * unique_threats)
    
    # Keyword density and severity (25%)
    total_keywords = sum(len(keywords) for keywords in label_keywords.values())
    found_keywords = sum(1 for keywords in label_keywords.values() for kw in keywords if kw in message)
    keyword_density = found_keywords / total_keywords if total_keywords > 0 else 0
    
    # Check for critical keywords that should boost the score
    critical_bonus = 0.25 if any(kw in message for kw in critical_keywords) else 0.0
    
    # Check for links in the message (25% bonus)
    link_bonus = 0.25 if "http://" in message or "https://" in message else 0.0
    
    # Apply severity multipliers based on threat type
    severity_multiplier = 1.0
    if isinstance(row['labels'], list):
        for label in row['labels']:
            if label in THREAT_SEVERITY_WEIGHTS:
                severity_multiplier = max(severity_multiplier, THREAT_SEVERITY_WEIGHTS[label])
        
        # Additional multiplier for critical threat combinations
        critical_combinations = [
            {"ransomware_with_data_theft", "data_exfiltration"},
            {"wiper", "data_exfiltration"},
            {"ransomware_with_data_theft", "wiper"},
            {"ddos", "data_exfiltration"},
            {"ransomware_with_data_theft", "ddos"},
            {"wiper", "ddos"}
        ]
        labels_set = set(row['labels'])
        if any(combo.issubset(labels_set) for combo in critical_combinations):
            severity_multiplier *= 1.35
    
    score = min(1.0, (base_score + threat_diversity + (0.25 * keyword_density) + critical_bonus + link_bonus) * severity_multiplier)
    
    # Ensure minimum score for critical threats
    if has_technical_pattern or any(indicator in message for indicator in critical_indicators):
        score = max(score, 0.55)
    
    return score

def calculate_engagement_score(row):
    """Calculate engagement score (30%)."""
    # Base engagement score
    engagement_score = 0.0
    
    # Forwards (20%)
    if row['forwards'] and row['forwards'] > 0:
        # More aggressive logarithmic scale for forwards
        forwards_score = min(1.0, math.log1p(row['forwards']) / math.log1p(25))  # Reduced threshold from 50 to 25
        engagement_score += 0.2 * forwards_score
    
    # Replies (10%)
    if row['reply_count'] and row['reply_count'] > 0:
        # More aggressive logarithmic scale for replies
        replies_score = min(1.0, math.log1p(row['reply_count']) / math.log1p(15))  # Reduced threshold from 25 to 15
        engagement_score += 0.1 * replies_score
    
    return engagement_score

def calculate_context_quality(row):
    """Calculate context quality score (20%)."""
    if not isinstance(row['text'], str):
        return 0.0
        
    message = row['text'].lower()
    
    # Message completeness (10%)
    completeness_score = 0.0
    elements_found = 0
    for element, keywords in KEY_ELEMENTS.items():
        if any(kw in message for kw in keywords):
            elements_found += 1
            completeness_score += 0.25  # Each element found adds 0.25
    
    # Bonus for having all elements (5%)
    if elements_found >= len(KEY_ELEMENTS):
        completeness_score += 0.05
    
    # Source credibility (10%)
    credibility_score = 0.6  # Increased default value
    if row['channel_username'] in ['thehackernews', 'bleepingcomputer', 'cveNotify']:
        credibility_score = 0.9
    elif row['channel_username'] in ['cybersecurityexperts', 'MalwareResearch']:
        credibility_score = 1.0
    
    return min(1.0, (0.1 * completeness_score) + (0.1 * credibility_score))

def calculate_temporal_relevance(row):
    """Calculate temporal relevance score (10%)."""
    if not isinstance(row['timestamp'], str):
        return 0.0
        
    # Recency (10%)
    try:
        message_time = pd.to_datetime(row['timestamp'])
        current_time = pd.Timestamp.now()
        hours_old = (current_time - message_time).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_old / 168))  # Decay over 1 week
    except:
        recency_score = 0.5
    
    return 0.1 * recency_score

def calculate_usefulness_score(row):
    """Calculate the overall usefulness score."""
    # Only calculate score for labeled messages
    if not isinstance(row['labels'], list) or len(row['labels']) == 0:
        return 0.0
    
    threat_relevance = calculate_threat_relevance(row)
    engagement_score = calculate_engagement_score(row)
    context_quality = calculate_context_quality(row)
    
    # Calculate base score with higher weight on threat relevance
    usefulness = (0.7 * threat_relevance) + \
                 (0.2 * engagement_score) + \
                 (0.1 * context_quality)
    
    # Bonus for messages with high threat relevance AND high engagement
    if threat_relevance >= 0.4 and engagement_score >= 0.3:
        usefulness += 0.2  # Additional 20% bonus
    
    # Ensure minimum score for critical threats
    message = row['text'].lower() if isinstance(row['text'], str) else ""
    critical_indicators = [
        "zero-day", "0day", "critical vulnerability", "active exploit",
        "mass attack", "widespread attack", "under attack", "actively exploited",
        "ransomware attack", "data breach", "massive breach", "critical severity"
    ]
    if any(indicator in message for indicator in critical_indicators):
        usefulness = max(usefulness, 0.5)  # Ensure at least HIGH classification
    
    # Round to 4 decimal places
    usefulness = round(usefulness, 4)
    
    # Cap at 1.0
    return min(1.0, usefulness)

def get_criticality_level(score):
    """Determine the criticality level based on the score."""
    score = round(score, 4)  # Round to 4 decimal places
    for level, threshold in CRITICALITY_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "INFO"

def main():
    # Read the scraped messages
    df = pd.read_csv("telegram_messages.csv")
    
    # Apply classification and scoring
    df['labels'] = df['text'].apply(classify_message)
    df['matched_keywords'] = df['text'].apply(get_matching_keywords)
    df['usefulness_score'] = df.apply(calculate_usefulness_score, axis=1)
    df['criticality'] = df['usefulness_score'].apply(get_criticality_level)
    
    # Save to CSV
    csv_filename = "telegram_messages_classified.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved classified messages to {csv_filename}")
    
    # Create and save filtered version with only labeled messages
    labeled_df = df[df['labels'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    labeled_df = labeled_df.sort_values('usefulness_score', ascending=False)
    
    # Save labeled messages to CSV
    labeled_csv_filename = "telegram_messages_labeled.csv"
    labeled_df.to_csv(labeled_csv_filename, index=False)
    print(f"Saved {len(labeled_df)} labeled messages to {labeled_csv_filename}")
    
    # Save labeled messages to Excel with formatting
    labeled_excel_filename = "telegram_messages_labeled.xlsx"
    writer = pd.ExcelWriter(labeled_excel_filename, engine='xlsxwriter')
    labeled_df.to_excel(writer, sheet_name='Labeled Messages', index=False)
    
    # Get the workbook and worksheet objects for labeled messages
    workbook = writer.book
    worksheet = writer.sheets['Labeled Messages']
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1
    })
    
    # Define criticality colors
    criticality_colors = {
        "CRITICAL": "#FF0000",  # Red
        "HIGH": "#FF6B6B",      # Light Red
        "MEDIUM": "#FFD93D",    # Yellow
        "LOW": "#6BCB77",       # Light Green
        "INFO": "#4D96FF"       # Blue
    }
    
    # Create formats for each criticality level
    criticality_formats = {}
    for level, color in criticality_colors.items():
        criticality_formats[level] = workbook.add_format({
            'bg_color': color,
            'font_color': '#FFFFFF' if level in ["CRITICAL", "HIGH"] else '#000000',
            'bold': True
        })
    
    # Format the header
    for col_num, value in enumerate(labeled_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # Set column widths
    worksheet.set_column('A:A', 20)  # Channel name
    worksheet.set_column('B:B', 20)  # Channel username
    worksheet.set_column('C:C', 10)  # Message ID
    worksheet.set_column('D:D', 20)  # Timestamp
    worksheet.set_column('E:E', 50)  # Message text
    worksheet.set_column('F:F', 15)  # Forwards
    worksheet.set_column('G:G', 15)  # Reply count
    worksheet.set_column('H:H', 30)  # Labels
    worksheet.set_column('I:I', 30)  # Matched keywords
    worksheet.set_column('J:J', 15)  # Usefulness score
    worksheet.set_column('K:K', 15)  # Criticality
    
    # Add conditional formatting for usefulness score
    worksheet.conditional_format('J2:J' + str(len(labeled_df) + 1), {
        'type': '3_color_scale',
        'min_color': "#FF0000",
        'mid_color': "#FFFF00",
        'max_color': "#00FF00"
    })
    
    # Apply criticality formatting
    for row in range(1, len(labeled_df) + 1):
        criticality = labeled_df.iloc[row-1]['criticality']
        worksheet.write(row, 10, criticality, criticality_formats[criticality])
    
    # Freeze the header row
    worksheet.freeze_panes(1, 0)
    
    # Add autofilter
    worksheet.autofilter(0, 0, len(labeled_df), len(labeled_df.columns) - 1)
    
    # Save the labeled Excel file
    writer.close()
    print(f"Saved formatted labeled messages to {labeled_excel_filename}")
    
    # Save complete dataset to Excel with formatting
    excel_filename = "telegram_messages_classified.xlsx"
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Messages', index=False)
    
    # Get the workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Messages']
    
    # Format the header
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # Set column widths
    worksheet.set_column('A:A', 20)  # Channel name
    worksheet.set_column('B:B', 20)  # Channel username
    worksheet.set_column('C:C', 10)  # Message ID
    worksheet.set_column('D:D', 20)  # Timestamp
    worksheet.set_column('E:E', 50)  # Message text
    worksheet.set_column('F:F', 15)  # Forwards
    worksheet.set_column('G:G', 15)  # Reply count
    worksheet.set_column('H:H', 30)  # Labels
    worksheet.set_column('I:I', 30)  # Matched keywords
    worksheet.set_column('J:J', 15)  # Usefulness score
    worksheet.set_column('K:K', 15)  # Criticality
    
    # Add conditional formatting for usefulness score
    worksheet.conditional_format('J2:J' + str(len(df) + 1), {
        'type': '3_color_scale',
        'min_color': "#FF0000",
        'mid_color': "#FFFF00",
        'max_color': "#00FF00"
    })
    
    # Apply criticality formatting
    for row in range(1, len(df) + 1):
        criticality = df.iloc[row-1]['criticality']
        worksheet.write(row, 10, criticality, criticality_formats[criticality])
    
    # Freeze the header row
    worksheet.freeze_panes(1, 0)
    
    # Add autofilter
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
    
    # Save the complete Excel file
    writer.close()
    print(f"Saved formatted complete messages to {excel_filename}")

if __name__ == "__main__":
    main() 