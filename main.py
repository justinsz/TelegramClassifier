from telethon.sync import TelegramClient
from telethon.tl.types import PeerChannel
import pandas as pd
import asyncio

# === Configuration ===
api_id = YOUR_API_ID              # Replace with your API ID
api_hash = 'YOUR_API_HASH'        # Replace with your API Hash
phone = 'YOUR_PHONE_NUMBER'       # Optional: Needed for first login only

# List of channels to scrape
channels = [
    "@CyberWatch", "@ThreatFeedDaily", "@ZeroDayAlerts",
    "@DarkNetLeaks", "@InfosecNews"
    # Add more here...
]

max_messages = 100

# === Main Script ===
data = []

async def main():
    async with TelegramClient('session_name', api_id, api_hash) as client:
        for channel in channels:
            try:
                async for message in client.iter_messages(channel, limit=max_messages):
                    if message.message:
                        data.append({
                            "channel": channel,
                            "message_id": message.id,
                            "timestamp": message.date,
                            "text": message.message,
                            "forwards": message.forwards,
                            "reply_count": getattr(message, 'replies', None).replies if message.replies else None
                        })
            except Exception as e:
                print(f"Error scraping {channel}: {e}")

        # Export to CSV
        df = pd.DataFrame(data)
        df.to_csv("telegram_messages.csv", index=False)
        print("Saved to telegram_messages.csv")

if __name__ == '__main__':
    asyncio.run(main())
