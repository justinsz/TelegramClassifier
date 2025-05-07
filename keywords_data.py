# Keyword lists with expanded cybersecurity-specific terms
label_keywords = {
    "cve": [
        "cve-", "common vulnerabilities and exposures", "vulnerability identifier"
    ],
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

# Threat severity weights for different categories
THREAT_SEVERITY_WEIGHTS = {
    "cve": 0.9,  # High severity for CVEs
    "ransomware_with_data_theft": 1.0,
    "wiper": 1.0,
    "ransomware_no_data_theft": 0.9,
    "data_exfiltration": 0.9,
    "ddos": 0.8,
    "fraud": 0.7,
    "defacement": 0.6
}

# Criticality thresholds for classification
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
    r"CVE-\d{4}-\d{4,7}",  # Basic CVE pattern
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

# Keywords and patterns for identifying informational content
INFO_KEYWORDS = [
    "guide", "tutorial", "how to", "learn", "course", "training",
    "basics", "introduction", "overview", "fundamentals",
    "best practices", "tips", "tricks", "recommendations",
    "checklist", "resources", "tools", "software", "top 10",
    "walkthrough", "step by step", "explained", "understanding"
]

INFO_PATTERNS = [
    r"^\d+\s+ways\s+to",
    r"^\d+\s+tips\s+for",
    r"^\d+\s+best\s+practices",
    r"how\s+to\s+\w+",
    r"learn\s+\w+",
    r"guide\s+to\s+\w+",
    r"introduction\s+to\s+\w+",
    r"basics\s+of\s+\w+"
]

# Indicators for user-generated content
USER_CONTENT_INDICATORS = [
    r"my experience",
    r"i learned",
    r"i found",
    r"i discovered",
    r"i created",
    r"i made",
    r"i built",
    r"i developed",
    r"i wrote",
    r"i'm sharing",
    r"sharing my",
    r"check out my",
    r"follow me",
    r"subscribe"
] 