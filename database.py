"""SQLite schema and connection handling for SceneIt.

Deliberately imports nothing from `config`, so this module stays usable without
pulling in torch.
"""

import os
import sqlite3

DB_PATH = os.getenv("DB_PATH", "sceneit.db")

# Bumped whenever the schema changes in a way that needs migrating.
SCHEMA_VERSION = 5


def _ensure_columns(conn):
    """Add any columns missing from an older database.

    Called by both init_db() and migrate(). CREATE TABLE IF NOT EXISTS is a
    no-op on an existing table, so a v2 database kept its original `videos`
    definition while init_db() went on to index videos(status) -- a column that
    did not exist yet. Schema evolution has to happen before anything depends
    on the new shape.
    """
    vcols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)")}
    if vcols:
        for col, decl in [
            ("source_url", "TEXT"),
            ("source_type", "TEXT"),
            ("synopsis", "TEXT"),
            ("duration_sec", "REAL"),
            ("status", "TEXT DEFAULT 'pending'"),
            ("error", "TEXT"),
            ("added_at", "TEXT"),
            ("ingested_at", "TEXT"),
        ]:
            if col not in vcols:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {decl}")
        conn.execute(
            "UPDATE videos SET status = 'ingested' WHERE status IS NULL AND id IN "
            "(SELECT DISTINCT video_id FROM frames)"
        )
        conn.execute("UPDATE videos SET status = 'pending' WHERE status IS NULL")

    acols = {r["name"] for r in conn.execute("PRAGMA table_info(actors)")}
    if acols:
        for col, decl in [("tmdb_id", "INTEGER"), ("profile_path", "TEXT"), ("source", "TEXT")]:
            if col not in acols:
                conn.execute(f"ALTER TABLE actors ADD COLUMN {col} {decl}")

    facols = {r["name"] for r in conn.execute("PRAGMA table_info(frame_actors)")}
    if facols and "confidence" not in facols:
        # The old table was keyed (frame_id, actor_id) with no evidence columns,
        # so it could not represent two faces in one frame. Rebuild it.
        conn.execute("ALTER TABLE frame_actors RENAME TO frame_actors_v3")
        conn.execute("""
            CREATE TABLE frame_actors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                actor_id INTEGER,
                cluster_id INTEGER,
                confidence REAL,
                method TEXT,
                bbox TEXT,
                det_score REAL,
                FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE,
                FOREIGN KEY(actor_id) REFERENCES actors(id) ON DELETE SET NULL
            )
        """)
        conn.execute(
            "INSERT INTO frame_actors (frame_id, actor_id, method) "
            "SELECT frame_id, actor_id, 'legacy' FROM frame_actors_v3"
        )
        conn.execute("DROP TABLE frame_actors_v3")

    fcols = {r["name"] for r in conn.execute("PRAGMA table_info(frames)")}
    for col in ("caption", "caption_model"):
        if fcols and col not in fcols:
            conn.execute(f"ALTER TABLE frames ADD COLUMN {col} TEXT")
    if fcols and "vector_id" not in fcols:
        conn.execute("ALTER TABLE frames ADD COLUMN vector_id INTEGER")
        # Backfilled from the old positional assumption, which is exactly what
        # can no longer be trusted -- rebuild the index rather than rely on it.
        conn.execute("UPDATE frames SET vector_id = id - 1 WHERE vector_id IS NULL")


def get_db_connection(db_path=None):
    """Open a connection with row access by name and foreign keys enforced."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()

    # The library. Previously this table was populated from a hardcoded Python
    # list in config.py on every extraction run, so the set of searchable films
    # was a code change rather than data. Sources are now rows: added through
    # the CLI or POST /api/library, and ingested independently.
    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE NOT NULL,
            filename TEXT,
            year TEXT,
            source_url TEXT,
            source_type TEXT,          -- 'http' | 'youtube' | 'local'
            synopsis TEXT,
            duration_sec REAL,
            status TEXT DEFAULT 'pending',   -- 'pending' | 'ingested' | 'failed'
            error TEXT,
            added_at TEXT,
            ingested_at TEXT,
            metadata_json TEXT         -- streaming links, TMDB payload, extras
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
            caption TEXT,              -- VLM description, generated at ingest
            caption_model TEXT,        -- which model wrote it, for re-runs
            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS actors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            character TEXT,
            image TEXT,              -- servable URL, never a filesystem path
            tmdb_id INTEGER,
            profile_path TEXT,       -- TMDB portrait path, for refetching
            face_encoding BLOB,      -- float32 reference embedding
            source TEXT              -- 'tmdb' | 'manual' | 'discovered'
        )
    """)

    # One row per DETECTED FACE, not per (frame, actor) pair. A frame can hold
    # two faces of the same person, and we need to know where each one is and
    # how confident the attribution was -- so the primary key is the face, and
    # the evidence travels with it.
    c.execute("""
        CREATE TABLE IF NOT EXISTS frame_actors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER NOT NULL,
            actor_id INTEGER,            -- NULL = face found but not identified
            cluster_id INTEGER,          -- identity group it belongs to
            confidence REAL,             -- cosine similarity behind the call
            method TEXT,                 -- 'cluster' | 'direct' | 'manual'
            bbox TEXT,                   -- "x1,y1,x2,y2" within the frame
            det_score REAL,
            FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE,
            FOREIGN KEY(actor_id) REFERENCES actors(id) ON DELETE SET NULL
        )
    """)

    # Bring an older table definition up to date before anything indexes it.
    _ensure_columns(conn)

    # Indexes for the hot paths: filtering frames by title, and resolving a
    # vector id back to a frame after a search.
    c.execute("CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frames_vector ON frames(vector_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frame_actors_frame ON frame_actors(frame_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frame_actors_actor ON frame_actors(actor_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_frame_actors_cluster ON frame_actors(cluster_id)")

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

    _ensure_columns(conn)
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
