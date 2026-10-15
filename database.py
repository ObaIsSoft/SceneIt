"""SQLite schema and connection handling for SceneIt.

Deliberately imports nothing from `config`, so this module stays usable without
pulling in torch.
"""

import os
import sqlite3

DB_PATH = os.getenv("DB_PATH", "sceneit.db")

# Bumped whenever the schema changes in a way that needs migrating.
SCHEMA_VERSION = 2


def get_db_connection(db_path=None):
    """Open a connection with row access by name and foreign keys enforced."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE NOT NULL,
            filename TEXT,
            year TEXT,
            metadata_json TEXT
        )
    """)

    # `vector_id` is the crux of this schema.
    #
    # Previously the link between a frame and its embedding was *positional*:
    # FAISS row N was assumed to be the Nth row of `SELECT ... ORDER BY id`.
    # Nothing enforced that. Any partial re-ingest, any frame inserted without
    # a matching vector, any deletion, shifted every subsequent frame by one and
    # the app then returned confidently wrong results -- the wrong film, the
    # wrong timestamp -- with no error anywhere.
    #
    # The id is now stored explicitly and handed to faiss.IndexIDMap2, so a
    # search returns these ids directly. Alignment is a property of the data
    # rather than an assumption about insertion order. NULL means "this frame
    # has no embedding yet", which is a legal, representable state.
    c.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            filename TEXT UNIQUE NOT NULL,
            timestamp TEXT,
            time_sec REAL,
            vector_id INTEGER UNIQUE,
            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS actors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            character TEXT,
            image TEXT,
            face_encoding BLOB
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS frame_actors (
            frame_id INTEGER NOT NULL,
            actor_id INTEGER NOT NULL,
            PRIMARY KEY (frame_id, actor_id),
            FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE,
            FOREIGN KEY(actor_id) REFERENCES actors(id) ON DELETE CASCADE
        )
    """)

    # Indexes for the hot paths: filtering frames by title, and resolving a
    # vector id back to a frame after a search.
    c.execute("CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frames_vector ON frames(vector_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frame_actors_frame ON frame_actors(frame_id)")

    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    conn.close()


def migrate(db_path=None):
    """Bring an existing database up to SCHEMA_VERSION.

    Idempotent, so it is safe to call on every startup.
    """
    conn = get_db_connection(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]

    if version >= SCHEMA_VERSION:
        conn.close()
        return version, False

    # v1 -> v2: add the explicit vector id.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(frames)")}
    if cols and "vector_id" not in cols:
        conn.execute("ALTER TABLE frames ADD COLUMN vector_id INTEGER")
        # Backfill from the old positional assumption. This is only correct if
        # the index was never partially rebuilt, which is exactly what we can no
        # longer assume -- so the index should be rebuilt rather than trusted.
        conn.execute("UPDATE frames SET vector_id = id - 1 WHERE vector_id IS NULL")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_frames_vector_unique ON frames(vector_id)"
        )

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    conn.close()
    return version, True


if __name__ == "__main__":
    init_db()
    print(f"--- Database initialised at {DB_PATH} (schema v{SCHEMA_VERSION}) ---")
