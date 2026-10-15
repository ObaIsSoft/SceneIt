import json
import sqlite3
import config
from database import get_db_connection

def migrate():
    print("--- Starting Migration (JSON -> SQLite) ---")
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Migrate Config Videos -> 'videos' table
    print("Migrating Video Config...")
    video_map = {} # title -> id
    
    for vid in config.VIDEOS_TO_PROCESS:
        # Check if exists
        c.execute("SELECT id FROM videos WHERE title = ?", (vid['title'],))
        existing = c.fetchone()
        
        metadata = {
            "synopsis": vid.get("synopsis"),
            "actors": vid.get("actors"),
            "streaming_on": vid.get("streaming_on")
        }
        
        if existing:
            print(f"Skipping {vid['title']} (already exists)")
            video_map[vid['title']] = existing['id']
        else:
            c.execute('''
                INSERT INTO videos (title, filename, year, metadata_json)
                VALUES (?, ?, ?, ?)
            ''', (
                vid['title'], 
                vid['filename'], 
                vid['year'], 
                json.dumps(metadata)
            ))
            video_map[vid['title']] = c.lastrowid
            print(f"Inserted {vid['title']}")
            
    conn.commit()
    
    # 2. Migrate Frames (metadata.json) -> 'frames' table
    if not os.path.exists(config.METADATA_JSON):
        print("No metadata.json found. Skipping frames migration.")
        return

    print(f"Loading {config.METADATA_JSON}...")
    with open(config.METADATA_JSON, 'r') as f:
        frames = json.load(f)
        
    print(f"Migrating {len(frames)} frames...")
    count = 0
    skipped = 0
    
    for frame in frames:
        video_title = frame.get('video_title')
        video_id = video_map.get(video_title)
        
        if not video_id:
            # Fallback: if we have frames for a video not in config anymore?
            # Create a dummy entry or skip? Skip for now.
            # print(f"Warning: Unknown video title {video_title}")
            continue
            
        # Check uniqueness
        try:
            c.execute('''
                INSERT INTO frames (video_id, filename, timestamp, time_sec)
                VALUES (?, ?, ?, ?)
            ''', (
                video_id,
                frame['filename'],
                frame['timestamp'],
                frame['time_sec']
            ))
            count += 1
        except sqlite3.IntegrityError:
            skipped += 1
            
    conn.commit()
    conn.close()
    print(f"--- Migration Complete: {count} frames inserted, {skipped} skipped. ---")

if __name__ == "__main__":
    import os
    migrate()
