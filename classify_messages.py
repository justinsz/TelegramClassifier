import pandas as pd
import numpy as np
from datetime import datetime
import math
import re
from keywords_data import (
    label_keywords,
    THREAT_SEVERITY_WEIGHTS,
    CRITICALITY_THRESHOLDS,
    TECHNICAL_VULNERABILITIES,
    critical_keywords,
    CRITICAL_PATTERNS,
    KEY_ELEMENTS,
    INFO_KEYWORDS,
    INFO_PATTERNS,
    USER_CONTENT_INDICATORS
)

def is_informational_content(message):
    """Check if the message is informational rather than a threat report."""
    message = message.lower()
    
    # Check for informational keywords
    if any(keyword in message for keyword in INFO_KEYWORDS):
        return True
        
    # Check for informational patterns
    if any(re.search(pattern, message) for pattern in INFO_PATTERNS):
        return True
        
    # Check for user-generated content indicators
    if any(re.search(indicator, message) for indicator in USER_CONTENT_INDICATORS):
        return True
        
    return False

def classify_message(message: str):
    """Classify a message into one or more categories based on keywords."""
    if not isinstance(message, str):
        return []
    
    # Check for informational content first
    if is_informational_content(message):
        return []
    
    message = message.lower()
    labels = []

    # Check for CVE pattern first
    if re.search(r"cve-\d{4}-\d{4,7}", message, re.IGNORECASE):
        labels.append("cve")

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
    
    # Check for informational content first
    if is_informational_content(message):
        return 0.0
    
    # Check for guide/tutorial content first (automatic INFO classification)
    if "guide" in message or "tutorial" in message or "how to" in message:
        return 0.0
    
    # Check for CVE pattern first - automatically high severity
    if re.search(r"cve-\d{4}-\d{4,7}", message, re.IGNORECASE):
        base_score = 0.75  # Start with high base score for CVEs
    else:
        # Check for technical vulnerabilities
        base_score = 0.0
        for vuln, score in TECHNICAL_VULNERABILITIES.items():
            if vuln in message:
                base_score = max(base_score, score)
    
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
    """Calculate engagement score (20%)."""
    if not isinstance(row['forwards'], (int, float)) or not isinstance(row['reply_count'], (int, float)):
        return 0.0
    
    # Use a more aggressive logarithmic scale for forwards and replies
    forward_score = min(0.5, math.log1p(row['forwards']) / math.log1p(25))  # Reduced from 50 to 25
    reply_score = min(0.3, math.log1p(row['reply_count']) / math.log1p(15))  # Reduced from 25 to 15
    
    # Combine scores with weights
    return min(1.0, (forward_score * 0.7) + (reply_score * 0.3))

def calculate_usefulness_score(row):
    """Calculate final usefulness score."""
    threat_relevance = calculate_threat_relevance(row)
    engagement_score = calculate_engagement_score(row)
    
    # Increased weight for threat relevance (70%)
    # Reduced weight for engagement (20%)
    # Context quality is now implicit in threat relevance (10%)
    usefulness_score = (threat_relevance * 0.7) + (engagement_score * 0.2)
    
    # Additional bonus for messages with both high threat relevance and engagement
    if threat_relevance >= 0.5 and engagement_score >= 0.3:
        usefulness_score = min(1.0, usefulness_score + 0.1)
    
    return round(usefulness_score, 4)

def determine_criticality(score):
    """Determine criticality level based on score."""
    if score >= CRITICALITY_THRESHOLDS['CRITICAL']:
        return 'CRITICAL'
    elif score >= CRITICALITY_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif score >= CRITICALITY_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif score >= CRITICALITY_THRESHOLDS['LOW']:
        return 'LOW'
    else:
        return 'INFO'

def main():
    # Read the CSV file
    df = pd.read_csv('telegram_messages.csv')
    
    # Apply classification to each message
    print("Classifying messages...")
    df['labels'] = df['text'].apply(classify_message)
    df['matching_keywords'] = df['text'].apply(get_matching_keywords)
    
    # Calculate usefulness score
    print("Calculating usefulness scores...")
    df['usefulness_score'] = df.apply(calculate_usefulness_score, axis=1)
    
    # Determine criticality
    print("Determining criticality levels...")
    df['criticality'] = df['usefulness_score'].apply(determine_criticality)
    
    # Save the results
    print("Saving results...")
    df.to_csv('telegram_messages_classified.csv', index=False)
    print("Classification complete!")

if __name__ == '__main__':
    main() 