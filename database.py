import sqlite3
import os
import json
# import config # Removed to avoid torch dependency

DB_PATH = os.getenv("DB_PATH", "sceneit.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Videos Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE NOT NULL,
            filename TEXT,
            year TEXT,
            metadata_json TEXT -- Stores synopsis, actors list (legacy), streaming info
        )
    ''')
    
    # 2. Frames Table
    # Storing timestamp info and lookup keys. 
    # Note: Vector embedding is still in FAISS (index) for speed, linked by implicit row ID or explicit ID.
    # To keep it simple: We will load ALL frames from DB into memory 
    # and assume their order matches the FAISS index order (0..N).
    # OR better: Add an 'index_id' column to map explicitly.
    c.execute('''
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            filename TEXT UNIQUE NOT NULL, -- e.g. "TearsOfSteel_123.jpg"
            timestamp TEXT, -- Display string "0:05"
            time_sec REAL, -- For sorting
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
    ''')

    # 3. Actors Table (For Future Facial Recognition)
    c.execute('''
        CREATE TABLE IF NOT EXISTS actors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            character TEXT,
            image TEXT,
            face_encoding BLOB -- For future use
        )
    ''')

    # 4. Frame_Actors (Many-to-Many)
    c.execute('''
        CREATE TABLE IF NOT EXISTS frame_actors (
            frame_id INTEGER,
            actor_id INTEGER,
            FOREIGN KEY(frame_id) REFERENCES frames(id),
            FOREIGN KEY(actor_id) REFERENCES actors(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"--- Database Initialized at {DB_PATH} ---")

if __name__ == "__main__":
    init_db()
