import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
num_messages = 200
current_time = datetime.now()

data = []
threat_types = ['ddos', 'ransomware_with_data_theft', 'data_exfiltration', 'wiper', 'fraud', 'defacement', 'cve']
channels = ['CVE Notify', 'Cyber Security News', 'The Hacker News', 'BleepingComputer', 'Malware News', 'CTI Now']
severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']

for i in range(num_messages):
    timestamp = current_time - timedelta(hours=i*2)  # Messages spread over time
    num_labels = np.random.randint(1, 4)  # 1-3 labels per message
    labels = np.random.choice(threat_types, num_labels, replace=False).tolist()
    
    # Adjust probability based on number of labels
    base_critical_prob = 0.3 + (num_labels * 0.1)  # Increase CRITICAL probability with more labels
    remaining_prob = 1.0 - base_critical_prob
    severity_weights = [
        base_critical_prob,  # CRITICAL
        remaining_prob * 0.4,  # HIGH
        remaining_prob * 0.3,  # MEDIUM
        remaining_prob * 0.2,  # LOW
        remaining_prob * 0.1   # INFO
    ]
    
    severity = np.random.choice(severities, p=severity_weights)
    
    score = np.random.uniform(
        0.55 if severity == 'CRITICAL' else 0.35 if severity == 'HIGH' else 0.25 if severity == 'MEDIUM' else 0.15 if severity == 'LOW' else 0.0,
        1.0 if severity == 'CRITICAL' else 0.54 if severity == 'HIGH' else 0.34 if severity == 'MEDIUM' else 0.24 if severity == 'LOW' else 0.14
    )
    
    data.append({
        'channel': np.random.choice(channels),
        'channel_username': f"@{channels[np.random.randint(0, len(channels)-1)].lower().replace(' ', '_')}",
        'message_id': i + 1,
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'text': f"Sample threat alert {i+1} with severity {severity}",
        'forwards': np.random.randint(0, 100),
        'reply_count': np.random.randint(0, 50),
        'labels': str(labels),
        'matching_keywords': str(['sample', 'threat', severity.lower()]),
        'usefulness_score': score,
        'criticality': severity
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('telegram_messages_classified.csv', index=False)
print("Generated test data with", len(df), "messages") 