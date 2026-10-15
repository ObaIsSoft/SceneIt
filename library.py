"""The video library, as data.

Replaces `config.VIDEOS_TO_PROCESS` -- a 136-line Python literal that made
adding a film a code change, carried per-title branches keyed on
`if "Dune" in title`, and whose URLs have since all gone dead (every one of the
five Google CDN sample links now returns 403).

Sources are rows in the `videos` table. They are added through the CLI or
POST /api/library, then ingested separately, so discovery and extraction are
independent and a failed download does not abandon the whole run.

Imports nothing heavier than the standard library, so the CLI and the API can
manage the library without loading a model.
"""

import datetime
import json
import os
import re
from urllib.parse import urlparse

from database import get_db_connection

YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "youtu.be", "www.youtu.be", "music.youtube.com",
}

STATUS_PENDING = "pending"
STATUS_INGESTED = "ingested"
STATUS_FAILED = "failed"


# --- Classifying a source ---------------------------------------------------

def classify_source(source):
    """Return 'youtube', 'http' or 'local' for a URL or filesystem path."""
    if not source:
        return "local"
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return "youtube" if parsed.netloc.lower() in YOUTUBE_HOSTS else "http"
    return "local"


def youtube_id(url):
    """Extract a YouTube video id, or None.

    Handles watch?v=, youtu.be/, /shorts/ and /embed/. The old code did
    `url.split('v=')[1].split('&')[0]`, which raised IndexError on every form
    but the first.
    """
    if classify_source(url) != "youtube":
        return None
    parsed = urlparse(url)
    if parsed.netloc.lower().endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate or None
    for prefix in ("/shorts/", "/embed/", "/v/"):
        if parsed.path.startswith(prefix):
            candidate = parsed.path[len(prefix):].split("/")[0]
            return candidate or None
    for pair in parsed.query.split("&"):
        if pair.startswith("v="):
            return pair[2:] or None
    return None


def safe_filename(title, source=None, extension=".mp4"):
    """Build a filesystem-safe filename from a title.

    Keeps the result anchored to a basename so a crafted title cannot escape
    the media directory.
    """
    base = os.path.basename(urlparse(source or "").path) if source else ""
    if base and os.path.splitext(base)[1]:
        stem, extension = os.path.splitext(base)
    else:
        stem = title or "video"
    stem = re.sub(r"[^\w\s.-]", "", stem).strip()
    stem = re.sub(r"[\s_]+", "_", stem)[:80].strip("._-")
    return f"{stem or 'video'}{extension}"


def derive_title(source):
    """Best-effort title from a URL or path, used when none is supplied.

    YouTube is checked first: the path of a watch URL is just "/watch", so
    deriving from the path stem produced the title "watch" for every YouTube
    source. Callers normally pass a real title anyway -- discovery supplies the
    video's own -- so this is only the fallback.
    """
    if not source:
        return "Untitled"

    vid = youtube_id(source)
    if vid:
        return f"youtube:{vid}"

    stem = os.path.splitext(os.path.basename(urlparse(source).path))[0]
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem[:200] if stem else "Untitled"


# --- Reading ----------------------------------------------------------------

def _row_to_dict(row):
    data = dict(row)
    extras = {}
    if data.get("metadata_json"):
        try:
            extras = json.loads(data["metadata_json"]) or {}
        except (TypeError, ValueError):
            extras = {}
    data.pop("metadata_json", None)
    data["actors"] = extras.get("actors", [])
    data["streaming_on"] = extras.get("streaming_on", [])
    data["tmdb"] = extras.get("tmdb")
    return data


def list_sources(status=None, db_path=None):
    conn = get_db_connection(db_path)
    sql = (
        "SELECT v.*, (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id) AS frame_count "
        "FROM videos v"
    )
    params = ()
    if status:
        sql += " WHERE v.status = ?"
        params = (status,)
    sql += " ORDER BY v.id ASC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_source(identifier, db_path=None):
    """Look a source up by numeric id or exact title."""
    conn = get_db_connection(db_path)
    if isinstance(identifier, int) or str(identifier).isdigit():
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (int(identifier),)).fetchone()
    else:
        row = conn.execute("SELECT * FROM videos WHERE title = ?", (identifier,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def find_sources(terms, db_path=None):
    """Case-insensitive substring match across titles."""
    if not terms:
        return []
    wanted = [t.lower() for t in terms]
    return [s for s in list_sources(db_path=db_path)
            if any(w in (s["title"] or "").lower() for w in wanted)]


# --- Writing ----------------------------------------------------------------

def add_source(source, title=None, year=None, synopsis=None, metadata=None,
               db_path=None, replace=False):
    """Register a source. Returns (row_dict, created).

    `source` is a URL or a local path. Nothing is downloaded here; ingestion is
    a separate step, so adding is cheap and offline.
    """
    source = (source or "").strip()
    source_type = classify_source(source)

    if source_type == "local":
        if not source:
            raise ValueError("a local source needs a path")
        if not os.path.exists(source):
            raise FileNotFoundError(source)

    title = (title or derive_title(source)).strip()
    if not title:
        raise ValueError("a title is required")

    filename = safe_filename(title, source if source_type != "local" else None)
    if source_type == "local":
        filename = safe_filename(title, source)

    conn = get_db_connection(db_path)
    existing = conn.execute("SELECT * FROM videos WHERE title = ?", (title,)).fetchone()

    if existing and not replace:
        conn.close()
        return _row_to_dict(existing), False

    payload = dict(metadata or {})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if existing:
        conn.execute(
            "UPDATE videos SET filename = ?, year = ?, source_url = ?, source_type = ?, "
            "synopsis = ?, status = ?, error = NULL, metadata_json = ? WHERE id = ?",
            (filename, year, source, source_type, synopsis, STATUS_PENDING,
             json.dumps(payload), existing["id"]),
        )
        row_id = existing["id"]
    else:
        cur = conn.execute(
            "INSERT INTO videos (title, filename, year, source_url, source_type, synopsis, "
            "status, added_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (title, filename, year, source, source_type, synopsis,
             STATUS_PENDING, now, json.dumps(payload)),
        )
        row_id = cur.lastrowid

    conn.commit()
    row = conn.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return _row_to_dict(row), True


def remove_source(identifier, db_path=None):
    """Delete a source and, by cascade, its frames. Returns True if removed.

    The caller must rebuild the index afterwards: dropping frames leaves their
    vectors orphaned, which load_database() will report.
    """
    row = get_source(identifier, db_path=db_path)
    if not row:
        return False
    conn = get_db_connection(db_path)
    conn.execute("DELETE FROM videos WHERE id = ?", (row["id"],))
    conn.commit()
    conn.close()
    return True


def mark_ingested(video_id, duration_sec=None, db_path=None):
    conn = get_db_connection(db_path)
    conn.execute(
        "UPDATE videos SET status = ?, error = NULL, ingested_at = ?, "
        "duration_sec = COALESCE(?, duration_sec) WHERE id = ?",
        (STATUS_INGESTED, datetime.datetime.now(datetime.timezone.utc).isoformat(),
         duration_sec, video_id),
    )
    conn.commit()
    conn.close()


def mark_failed(video_id, error, db_path=None):
    conn = get_db_connection(db_path)
    conn.execute(
        "UPDATE videos SET status = ?, error = ? WHERE id = ?",
        (STATUS_FAILED, str(error)[:500], video_id),
    )
    conn.commit()
    conn.close()


def update_metadata(video_id, year=None, synopsis=None, actors=None,
                    streaming_on=None, tmdb=None, db_path=None):
    """Merge enrichment into a source without disturbing its ingest state."""
    conn = get_db_connection(db_path)
    row = conn.execute("SELECT metadata_json FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        conn.close()
        return False
    try:
        payload = json.loads(row["metadata_json"] or "{}") or {}
    except (TypeError, ValueError):
        payload = {}

    if actors is not None:
        payload["actors"] = actors
    if streaming_on is not None:
        payload["streaming_on"] = streaming_on
    if tmdb is not None:
        payload["tmdb"] = tmdb

    conn.execute(
        "UPDATE videos SET year = COALESCE(?, year), synopsis = COALESCE(?, synopsis), "
        "metadata_json = ? WHERE id = ?",
        (year, synopsis, json.dumps(payload), video_id),
    )
    conn.commit()
    conn.close()
    return True


# --- Serialising for the API ------------------------------------------------

def to_api_dict(source):
    """Shape a source for the frontend.

    Deliberately omits `source_url` and never emits a filesystem path: the old
    /api/config returned absolute paths like /Users/<name>/.../zendaya.jpg,
    which both leaked the server layout and rendered as broken images because
    the browser cannot fetch them.
    """
    return {
        "id": source["id"],
        "title": source["title"],
        "filename": source.get("filename"),
        "year": source.get("year") or "",
        "synopsis": source.get("synopsis") or "",
        "status": source.get("status"),
        "frame_count": source.get("frame_count", 0),
        "duration_sec": source.get("duration_sec"),
        "actors": source.get("actors") or [],
        "streaming_on": source.get("streaming_on") or [],
    }
