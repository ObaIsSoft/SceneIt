import os
import config
from googleapiclient.discovery import build

def test_api():
    print("Testing YouTube API...")
    key = getattr(config, 'YOUTUBE_API_KEY', None)
    if not key:
        print("FAIL: No API Key found in config.")
        return

    try:
        youtube = build('youtube', 'v3', developerKey=key)
        request = youtube.search().list(
            q="test",
            part="snippet",
            maxResults=1,
            type="video"
        )
        response = request.execute()
        print("SUCCESS: API call worked!")
        print(f"Found: {response['items'][0]['snippet']['title']}")
    except Exception as e:
        print(f"FAIL: API Error: {e}")

if __name__ == "__main__":
    test_api()
