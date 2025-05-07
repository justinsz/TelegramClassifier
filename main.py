from telethon.sync import TelegramClient
from telethon.tl.types import PeerChannel
import pandas as pd
import asyncio
from dotenv import load_dotenv
import os
from channels_data import channels, columns
import re
from telethon.errors import UsernameInvalidError, UsernameNotOccupiedError

# Load environment variables
load_dotenv()

# === Configuration ===
api_id = os.getenv('TELEGRAM_API_ID')
api_hash = os.getenv('TELEGRAM_API_HASH')
phone = os.getenv('TELEGRAM_PHONE')

max_messages = 100

def extract_channel_name(url):
    # Extract the channel name from the URL (what comes after t.me/)
    match = re.search(r't\.me/([^/]+)', url)
    if match:
        return match.group(1)
    return None

# === Main Script ===
data = []
successful_channels = []
failed_channels = []

async def main():
    async with TelegramClient('session_name', api_id, api_hash) as client:
        for channel_data in channels:
            channel_url = channel_data[2]  # Get the Telegram URL
            channel_name = extract_channel_name(channel_url)
            if not channel_name:
                print(f"Could not extract channel name from URL: {channel_url}")
                failed_channels.append((channel_data[1], "Invalid URL format"))
                continue
                
            try:
                print(f"Attempting to scrape channel: {channel_name}")
                message_count = 0
                async for message in client.iter_messages(channel_name, limit=max_messages):
                    if message.message:
                        data.append({
                            "channel": channel_data[1],  # Channel Name
                            "channel_username": channel_name,
                            "message_id": message.id,
                            "timestamp": message.date,
                            "text": message.message,
                            "forwards": message.forwards,
                            "reply_count": getattr(message, 'replies', None).replies if message.replies else None
                        })
                        message_count += 1
                
                if message_count > 0:
                    print(f"Successfully scraped {message_count} messages from {channel_name}")
                    successful_channels.append(channel_name)
                else:
                    print(f"No messages found for {channel_name}")
                    failed_channels.append((channel_name, "No messages found"))
                    
            except (UsernameInvalidError, UsernameNotOccupiedError) as e:
                print(f"Channel does not exist or is invalid: {channel_name}")
                failed_channels.append((channel_name, "Channel does not exist"))
            except Exception as e:
                print(f"Error scraping {channel_name}: {str(e)}")
                failed_channels.append((channel_name, str(e)))

        # Export to CSV
        if data:
            df = pd.DataFrame(data)
            df.to_csv("telegram_messages.csv", index=False)
            print(f"\nSuccessfully scraped {len(successful_channels)} channels")
            print(f"Saved {len(data)} messages to telegram_messages.csv")
        else:
            print("\nNo messages were scraped successfully")
            
        if failed_channels:
            print("\nFailed channels:")
            for channel, reason in failed_channels:
                print(f"- {channel}: {reason}")

if __name__ == '__main__':
    asyncio.run(main())
