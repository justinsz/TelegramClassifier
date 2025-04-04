# Channel data columns
columns = [
    "ID", "Channel Name", "Telegram URL", 
    "Credibility Score", "Credibility Category", 
    "Relevance Score", "Relevance Category", 
    "Engagement Score", "Engagement Category", 
    "Posting Frequency Score", "Frequency Category", 
    "Subscriber Count Score", "Audience Category"
]

# List of channels with their metadata
channels = [
    [1, "Cyber Security News", "https://t.me/cyber_security_channel", 9, "High", 10, "Very High", 6, "Moderate", 9, "Frequent", 9, "Large"],
    [2, "Cyber Security – InfoSec/IT Security Experts (Group)", "https://t.me/cybersecurityexperts", 8, "High", 9, "High", 8, "High", 8, "Frequent", 9, "Large"],
    [3, "Cloud & Cybersecurity", "https://t.me/cloudandcybersecurity", 8, "High", 8, "High", 7, "Moderate", 8, "Frequent", 8, "Large"],
    [4, "Cybersecurity & Privacy News", "https://t.me/cibsecurity", 9, "High", 9, "High", 5, "Medium", 9, "Frequent", 8, "Large"],
    [5, "Android Security & Malware", "https://t.me/androidMalware", 8, "High", 8, "High", 7, "Moderate", 8, "Frequent", 9, "Large"],
    [6, "Malware Research (Group)", "https://t.me/MalwareResearch", 9, "High", 9, "High", 8, "High", 7, "Regular", 7, "Medium"],
    [7, "BugCrowd (Bug Bounty Group)", "https://t.me/BugCrowd", 8, "High", 8, "High", 8, "High", 8, "Frequent", 8, "Large"],
    [8, "Red Team Alerts", "https://t.me/redteamalerts", 7, "High", 9, "Very High", 6, "Moderate", 9, "Frequent", 8, "Large"],
    [9, "Blue Team Alerts", "https://t.me/blueteamalerts", 7, "High", 9, "Very High", 6, "Moderate", 9, "Frequent", 7, "Medium"],
    [10, "IT Security Alerts", "https://t.me/itsecalert", 8, "High", 9, "High", 6, "Moderate", 9, "Frequent", 8, "Large"],
    [13, "CVE Notify", "https://t.me/cveNotify", 9, "High", 10, "Very High", 5, "Medium", 10, "Very Frequent", 8, "Large"],
    [14, "Malware News (malwr)", "https://t.me/malwr", 7, "Fair", 9, "High", 5, "Medium", 8, "Frequent", 8, "Large"],
    [15, "BleepingComputer News", "https://t.me/bleepingcomputer", 9, "High", 8, "High", 5, "Medium", 8, "Frequent", 7, "Medium"],
    [16, "The Hacker News (THN)", "https://t.me/thehackernews", 9, "High", 9, "High", 6, "Moderate", 9, "Frequent", 10, "Very Large"],
    [17, "Cyber Threat Intelligence (CTI Now)", "https://t.me/ctinow", 8, "High", 10, "Very High", 6, "Moderate", 8, "Frequent", 7, "Medium"],
    [18, "Learn Cybersecurity (TeamMatrix)", "https://t.me/teammatrixs", 7, "Good", 7, "High", 6, "Moderate", 7, "Regular", 6, "Medium"],
    [19, "Dark Web Informer (CTI)", "https://t.me/TheDarkWebInformer", 8, "High", 8, "High", 5, "Medium", 7, "Regular", 6, "Medium"],
    [20, "Hacker Exploits", "https://t.me/hackerexploits", 6, "Fair", 7, "High", 5, "Medium", 6, "Regular", 5, "Small"],
    [21, "Infosec Tutorials (Jazer)", "https://t.me/cybersecuriti", 6, "Fair", 6, "Moderate", 5, "Medium", 7, "Regular", 6, "Medium"],
    [22, "Cyber Ninja Sec Community", "https://t.me/Cyb3rn1nj4", 6, "Fair", 7, "High", 5, "Medium", 7, "Regular", 6, "Medium"],
    [23, "HackerSploit Channel", "https://t.me/Hackersploit", 8, "High", 6, "Moderate", 4, "Low", 5, "Low", 3, "Very Small"],
    [24, "Security Awareness", "https://t.me/SecurityAwareness", 7, "Good", 7, "High", 5, "Medium", 6, "Regular", 6, "Medium"],
    [25, "OSINT & Threats", "https://t.me/OSINT_threats", 7, "Good", 8, "High", 5, "Medium", 6, "Regular", 5, "Small"],
    [26, "Exploit Development", "https://t.me/ExploitDev", 7, "Good", 8, "High", 5, "Medium", 6, "Regular", 5, "Small"],
    [27, "Ransomware Alerts", "https://t.me/RansomwareAlerts", 8, "High", 9, "High", 5, "Medium", 7, "Regular", 6, "Medium"],
    [28, "Vuln Digest", "https://t.me/VulnDigest", 8, "High", 9, "High", 5, "Medium", 8, "Frequent", 6, "Medium"],
    [29, "Data Breach News", "https://t.me/DataBreachNews", 8, "High", 9, "High", 6, "Moderate", 8, "Frequent", 7, "Medium"],
    [30, "Threat Intel Updates", "https://t.me/ThreatIntel", 8, "High", 10, "Very High", 6, "Moderate", 8, "Frequent", 7, "Medium"],
    [31, "CyberSec Jobs & Alerts", "https://t.me/CyberSecurityJobs", 7, "Good", 6, "Moderate", 5, "Medium", 7, "Regular", 6, "Medium"],
    [32, "Threat Hunting & DFIR", "https://t.me/DFIRchat", 7, "Good", 8, "High", 6, "Moderate", 6, "Regular", 5, "Small"],
    [33, "Exploit News", "https://t.me/Exploit_News", 7, "Good", 9, "High", 5, "Medium", 7, "Regular", 6, "Medium"],
    [34, "Vulnerability Alert", "https://t.me/VulnerabilityAlert", 8, "High", 10, "Very High", 5, "Medium", 9, "Frequent", 7, "Medium"],
    [35, "Cybersecurity Magazine", "https://t.me/CyberSecMagazine", 7, "Good", 7, "High", 5, "Medium", 6, "Regular", 6, "Medium"],
    [36, "Hacker News Feed", "https://t.me/HNsecurity", 6, "Fair", 6, "Moderate", 4, "Low", 9, "Frequent", 6, "Medium"],
    [37, "Security Tool Updates", "https://t.me/SecTools", 7, "Good", 8, "High", 5, "Medium", 8, "Frequent", 6, "Medium"],
    [38, "Zero-Day Channel", "https://t.me/ZeroDayAlerts", 8, "High", 10, "Very High", 5, "Medium", 8, "Frequent", 6, "Medium"],
    [39, "Malware Analysis Hub", "https://t.me/MalwareAnalysisHub", 8, "High", 9, "High", 6, "Moderate", 7, "Regular", 6, "Medium"],
    [40, "SOC Threat Feeds", "https://t.me/SOCThreatFeeds", 7, "Good", 8, "High", 5, "Medium", 8, "Frequent", 5, "Small"],
    [41, "CyberSec Reddit Feed", "https://t.me/RedditCyberSec", 6, "Fair", 7, "High", 4, "Low", 9, "Frequent", 5, "Small"],
    [42, "Hackers Archive", "https://t.me/HackersArchive", 6, "Fair", 6, "Moderate", 4, "Low", 6, "Regular", 5, "Small"],
    [43, "Security Breach Monitor", "https://t.me/SecBreachMonitor", 7, "Good", 9, "High", 5, "Medium", 7, "Regular", 5, "Small"],
    [44, "Cyber Intelligence Wire", "https://t.me/CyberIntelWire", 7, "Good", 8, "High", 5, "Medium", 7, "Regular", 5, "Small"],
    [45, "Vulnerabilities Feed", "https://t.me/VulnerabilityFeed", 8, "High", 10, "Very High", 4, "Low", 10, "Very Frequent", 5, "Small"],
    [46, "Exploit Database Updates", "https://t.me/ExploitDB", 8, "High", 9, "High", 4, "Low", 9, "Frequent", 5, "Small"],
    [47, "INFOSEC Community", "https://t.me/InfoSecCommunity", 7, "Good", 7, "High", 6, "Moderate", 6, "Regular", 6, "Medium"],
    [48, "Threat Research Updates", "https://t.me/ThreatResearch", 8, "High", 8, "High", 5, "Medium", 6, "Regular", 5, "Small"],
    [49, "CyberSec Insights", "https://t.me/CyberSecInsights", 7, "Good", 8, "High", 5, "Medium", 6, "Regular", 5, "Small"],
    [50, "SOC Prime Threat Bounty", "https://t.me/SocPrimeThreat", 7, "Good", 8, "High", 5, "Medium", 5, "Low", 5, "Small"]
] 