"""
YouTube Integration for SceneIt
Fetches movie trailers and clips from YouTube for indexing.
"""

import os
import json
from typing import List, Dict, Optional
import subprocess

# YouTube Data API (will need: pip install google-api-python-client)
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Warning: google-api-python-client not installed. Run: pip install google-api-python-client")

import config

class YouTubeTrailerFetcher:
    """Fetches movie trailers from YouTube."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(config, 'YOUTUBE_API_KEY', None)
        self.youtube = None
        
        if self.api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            except Exception as e:
                print(f"Failed to initialize YouTube API: {e}")
    
    def search_trailers(self, query: str = "official movie trailer", max_results: int = 50) -> List[Dict]:
        """
        Search YouTube for movie trailers.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of video dictionaries with metadata
        """
        if not self.youtube:
            print("YouTube API not initialized. Please set YOUTUBE_API_KEY in config.")
            return []
        
        try:
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                type='video',
                maxResults=max_results,
                videoDuration='medium',  # 4-20 minutes (good for trailers)
                videoDefinition='high',   # HD only
                relevanceLanguage='en',
                safeSearch='moderate'
            ).execute()
            
            videos = []
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                videos.append({
                    'title': snippet['title'],
                    'video_id': video_id,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'thumbnail': snippet['thumbnails']['high']['url'],
                    'channel': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'],
                    'description': snippet.get('description', '')
                })
            
            return videos
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return []
    
    def download_video(self, video_url: str, output_folder: str, filename: str) -> Optional[str]:
        """
        Download YouTube video using yt-dlp.
        
        Args:
            video_url: YouTube video URL
            output_folder: Destination folder
            filename: Output filename (without extension)
            
        Returns:
            Path to downloaded file, or None if failed
        """
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{filename}.mp4")
        
        if os.path.exists(output_path):
            print(f"Video already exists: {filename}")
            return output_path
        
        try:
            # Using yt-dlp (better than youtube-dl)
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]',  # 720p max (faster processing)
                '--merge-output-format', 'mp4',
                '-o', output_path,
                video_url
            ]
            
            print(f"Downloading: {filename}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"✓ Downloaded: {filename}")
                return output_path
            else:
                print(f"✗ Download failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"✗ Download timeout: {filename}")
            return None
        except FileNotFoundError:
            print("✗ yt-dlp not found. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            print(f"✗ Download error: {e}")
            return None
    
    def get_popular_trailers(self, categories: List[str] = None) -> List[Dict]:
        """
        Get popular movie trailers from multiple search queries.
        
        Args:
            categories: List of search queries (defaults to current+classic movies)
            
        Returns:
            Combined list of video dictionaries
        """
        if categories is None:
            categories = [
                "official movie trailer 2024",
                "official movie trailer 2023",
                "blockbuster movie trailer",
                "new movie trailer",
                "classic movie trailer"
            ]
        
        all_videos = []
        seen_ids = set()
        
        for category in categories:
            print(f"Searching: {category}...")
            videos = self.search_trailers(category, max_results=10)
            
            # Deduplicate
            for video in videos:
                if video['video_id'] not in seen_ids:
                    all_videos.append(video)
                    seen_ids.add(video['video_id'])
            
        print(f"Found {len(all_videos)} unique trailers")
        return all_videos

def export_to_config_format(videos: List[Dict], output_file: str = "youtube_videos.json"):
    """
    Export video list to config-compatible format.
    
    Args:
        videos: List of video dictionaries from YouTubeTrailerFetcher
        output_file: JSON file to save to
    """
    formatted = []
    for video in videos:
        # Clean filename
        safe_title = "".join(c for c in video['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Max 50 chars
        
        formatted.append({
            "title": video['title'],
            "url": video['url'],
            "filename": f"{safe_title}.mp4",
            "source": "youtube",
            "video_id": video['video_id'],
            "channel": video['channel'],
            "thumbnail": video['thumbnail']
        })
    
    with open(output_file, 'w') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"Saved {len(formatted)} videos to {output_file}")
    return formatted

if __name__ == "__main__":
    # Demo usage
    print("=== SceneIt YouTube Trailer Fetcher ===\n")
    
    # Check for API key
    if not hasattr(config, 'YOUTUBE_API_KEY') or not config.YOUTUBE_API_KEY:
        print("ERROR: No YOUTUBE_API_KEY found in config.py")
        print("\nTo get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project")
        print("3. Enable YouTube Data API v3")
        print("4. Create credentials (API Key)")
        print("5. Add to config.py: YOUTUBE_API_KEY = 'your-key-here'")
        exit(1)
    
    fetcher = YouTubeTrailerFetcher()
    videos = fetcher.get_popular_trailers()
    
    if videos:
        export_to_config_format(videos)
        print("\n✓ Ready to download. Run build_trailer_database.py to proceed.")
