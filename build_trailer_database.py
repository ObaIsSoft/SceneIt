"""
Alternative: SceneIt Trailer Database Builder
Uses direct YouTube URLs instead of Search API.
"""

import os
import json
import sys
sys.path.append(os.path.dirname(__file__))

from youtube_integration import YouTubeTrailerFetcher
import config

# Curated list of popular movie trailer channels and specific videos
CURATED_TRAILERS = [
    # 2024 Blockbusters
    {"url": "https://www.youtube.com/watch?v=Way9Dexny3w", "title": "Dune Part Two Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=lV1OOlGwExM", "title": "Godzilla x Kong Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=73_1biulkYk", "title": "Deadpool & Wolverine Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=KBiOF3y1W0Y", "title": "The Fall Guy Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=qRuNxHF4AzQ", "title": "Beetlejuice Beetlejuice Official Trailer"},
   
    # 2023 Hits
    {"url": "https://www.youtube.com/watch?v=bK6ldnjE3Y0", "title": "Barbie Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=uYPbbksJxIg", "title": "Oppenheimer Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=AsWZ4E3btOE", "title": "Guardians of the Galaxy Vol 3 Trailer"},
    {"url": "https://www.youtube.com/watch?v=d9MyW72ELq0", "title": "Avatar The Way of Water Trailer"},
    {"url": "https://www.youtube.com/watch?v=giXco2jaZ_4", "title": "Spider-Man Across the Spider-Verse Trailer"},
    
    # Classics
    {"url": "https://www.youtube.com/watch?v=V75dMMIW2B4", "title": "Inception Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=w3ugHP-yZXw", "title": "The Dark Knight Trailer"},
    {"url": "https://www.youtube.com/watch?v=fFjdYrGZoYE", "title": "Interstellar Official Trailer"},
    {"url": "https://www.youtube.com/watch?v=HhesaQXLuRY", "title": "The Avengers Trailer"},
    {"url": "https://www.youtube.com/watch?v=sY1S34973zA", "title": "Joker Official Trailer"},
]

def build_database(max_videos=15, download_videos=True):
    """
    Build SceneIt trailer database from curated list.
    
    Args:
        max_videos: Maximum number of trailers to process
        download_videos: Whether to download videos (False for testing)
    """
    print("=== SceneIt Trailer Database Builder ===\n")
    
    fetcher = YouTubeTrailerFetcher()
    processed_videos = []
    
    # 1. Fetch from YouTube API (Auto-Discovery)
    api_videos = []
    if fetcher.youtube and getattr(config, 'YOUTUBE_SEARCH_QUERIES', None):
        print("--- API Enabled: Searching for trending trailers... ---")
        try:
            api_results = fetcher.get_popular_trailers(config.YOUTUBE_SEARCH_QUERIES)
            # Convert to internal format
            from youtube_integration import export_to_config_format
            # We use a temporary export to get the list format, simpler to just map here
            for v in api_results:
                 # Create clean filename
                safe_title = "".join(c for c in v['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]
                
                api_videos.append({
                    "title": v['title'],
                    "url": v['url'],
                    "filename": f"{safe_title}.mp4",
                    "source": "youtube",
                    "video_id": v['video_id'],
                    "thumbnail": v['thumbnail'],
                    "synopsis": v.get('description', '')[:200] + "..."
                })
            print(f"--- Discovered {len(api_videos)} trailers via API ---")
        except Exception as e:
            print(f"API Search failed: {e}")

    # 2. Combine with Curated (Curated takes priority for duplicates?) 
    # Actually, let's just merge.
    all_candidates = CURATED_TRAILERS + api_videos
    
    # Deduplicate by URL or Title
    seen_urls = set()
    final_list = []
    for cand in all_candidates:
        if cand['url'] not in seen_urls:
            final_list.append(cand)
            seen_urls.add(cand['url'])
            
    # Process
    for i, trailer in enumerate(final_list[:max_videos]):
        print(f"\n[{i+1}/{min(max_videos, len(final_list))}] Processing: {trailer['title']}")
        
        # Extract video ID from URL
        video_id = trailer['url'].split('v=')[1].split('&')[0]
        
        # Create clean filename
        safe_title = "".join(c for c in trailer['title'] if c.isalnum() or c in (' ', '-')).strip()
        safe_title = safe_title.replace(' ', '_')[:40]
        
        video_data = {
            "title": trailer['title'],
            "url": trailer['url'],
            "filename": f"{safe_title}.mp4",
            "source": "youtube",
            "video_id": video_id
        }
        
        if download_videos:
            # Download the video
            output_path = fetcher.download_video(
                trailer['url'],
                config.TEMP_VIDEO_DIR,
                safe_title
            )
            
            if output_path:
                processed_videos.append(video_data)
                print(f"  ✓ Ready for indexing")
            else:
                print(f"  ✗ Download failed, skipping")
        else:
            processed_videos.append(video_data)
            print(f"  → Added to list (download skipped)")
    
    # Save to config-compatible format
    output_file = "sceneit_trailers.json"
    with open(output_file, 'w') as f:
        json.dump(processed_videos, f, indent=2)
    
    print(f"\n✓ Saved {len(processed_videos)} trailers to {output_file}")
    print(f"\nNext steps:")
    print(f"1. Update config.py to use these videos")
    print(f"2. Run: python app.py --extract")
    print(f"3. Launch SceneIt!")
    
    return processed_videos

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SceneIt trailer database")
    parser.add_argument('--count', type=int, default=15, help="Number of trailers to process")
    parser.add_argument('--no-download', action='store_true', help="Skip downloading (list only)")
    
    args = parser.parse_args()
    
    build_database(max_videos=args.count, download_videos=not args.no_download)
