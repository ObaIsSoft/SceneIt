"""Shared fixtures.

Design rules for this suite:

* No network. Nothing here downloads a video, a model or a metadata record.
* No model by default. CLIP takes seconds to load and tens of seconds to run,
  so tests that only exercise index wiring inject synthetic unit vectors
  instead. Tests that genuinely need the encoder are marked `slow`.
* No writes outside tmp_path. The previous suite POSTed to /api/feedback and
  appended to the real feedback_log.json, so running the tests mutated
  production data.
"""

import json

import numpy as np
import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the database module at a throwaway file and initialise it."""
    import database

    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db(str(db_file))
    return str(db_file)


@pytest.fixture
def unit_vectors():
    """Deterministic L2-normalised vectors, as the encoder would produce."""

    def _make(n, dim=512, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, dim)).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    return _make


@pytest.fixture
def populated_library(temp_db, unit_vectors):
    """A small library: 2 videos, 6 frames, 6 vectors, ids written to the rows.

    Returns the connection details plus the vectors, so a test can query with a
    known frame's own vector and assert it gets that exact frame back.
    """
    import database

    frames = [
        ("Alpha", "Alpha_0.jpg", "0:00", 0.0),
        ("Alpha", "Alpha_1000.jpg", "0:01", 1.0),
        ("Alpha", "Alpha_2000.jpg", "0:02", 2.0),
        ("Beta", "Beta_0.jpg", "0:00", 0.0),
        ("Beta", "Beta_1000.jpg", "0:01", 1.0),
        ("Beta", "Beta_2000.jpg", "0:02", 2.0),
    ]
    vectors = unit_vectors(len(frames))

    conn = database.get_db_connection(temp_db)
    video_ids = {}
    for title in ("Alpha", "Beta"):
        cur = conn.execute(
            "INSERT INTO videos (title, filename, year, metadata_json) VALUES (?, ?, ?, ?)",
            (title, f"{title}.mp4", "2024", json.dumps({"synopsis": f"{title} synopsis"})),
        )
        video_ids[title] = cur.lastrowid

    for vector_id, (title, filename, ts, sec) in enumerate(frames):
        conn.execute(
            "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (video_ids[title], filename, ts, sec, vector_id),
        )
    conn.commit()
    conn.close()

    return {"db_path": temp_db, "frames": frames, "vectors": vectors}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Flask test client isolated from real data.

    The database is redirected too, not just the feedback file: /api/config and
    /api/library read the library, so without this the suite would read -- and
    POST /api/library would write to -- the developer's real sceneit.db.
    """
    import app as app_module
    import database

    db_file = tmp_path / "api.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db(str(db_file))

    feedback_file = tmp_path / "feedback.json"
    monkeypatch.setattr(app_module.config, "FEEDBACK_FILE", str(feedback_file))

    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c
