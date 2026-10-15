"""The tests that matter most: a search must return the frame it actually found.

The original design mapped FAISS row N to `metadata_db[N]` by list position.
Nothing enforced that correspondence, so any gap between the vector sequence and
the frame rows shifted every later frame by one and the app served a confidently
wrong film and timestamp with no error raised anywhere. These tests pin the
property that replaced it.
"""

import numpy as np
import pytest

from vector_index import VectorIndex

pytestmark = pytest.mark.unit


def build_index(vectors, ids):
    index = VectorIndex(vectors.shape[1])
    index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
    return index


def load_vector_map(db_path):
    """Mirror of SearchEngine.load_database's mapping step."""
    import database

    conn = database.get_db_connection(db_path)
    rows = conn.execute(
        "SELECT f.*, v.title AS video_title FROM frames f "
        "JOIN videos v ON f.video_id = v.id ORDER BY f.id ASC"
    ).fetchall()
    conn.close()
    return {r["vector_id"]: dict(r) for r in rows if r["vector_id"] is not None}


class TestIdentityLookup:
    def test_every_frame_retrieves_itself(self, populated_library):
        """Querying with a frame's own vector must return that frame."""
        vectors = populated_library["vectors"]
        vector_map = load_vector_map(populated_library["db_path"])
        index = build_index(vectors, list(range(len(vectors))))

        for expected_id, (_, filename, _, _) in enumerate(populated_library["frames"]):
            scores, ids = index.search(vectors[expected_id : expected_id + 1], 1)

            assert int(ids[0][0]) == expected_id
            assert vector_map[int(ids[0][0])]["filename"] == filename
            # Self-similarity of a unit vector is 1.0.
            assert scores[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_similarity_is_bounded(self, populated_library):
        """Inner product over unit vectors stays in [-1, 1], so it reads as a
        confidence without a conversion formula."""
        vectors = populated_library["vectors"]
        index = build_index(vectors, list(range(len(vectors))))
        scores, _ = index.search(vectors, len(vectors))

        assert scores.min() >= -1.0001
        assert scores.max() <= 1.0001


class TestRegressionPositionalDrift:
    """These are the cases the old positional mapping got wrong."""

    def test_deleted_frame_does_not_shift_later_frames(self, populated_library):
        """Remove a frame mid-library; the rest must still resolve correctly.

        Under positional mapping, deleting row 2 made every subsequent frame
        resolve one place early -- so a search would report the wrong timestamp,
        and often the wrong film.
        """
        import database

        db_path = populated_library["db_path"]
        vectors = populated_library["vectors"]

        conn = database.get_db_connection(db_path)
        conn.execute("DELETE FROM frames WHERE vector_id = 2")
        conn.commit()
        conn.close()

        vector_map = load_vector_map(db_path)
        # The index still holds all six vectors; the database now has five.
        index = build_index(vectors, list(range(len(vectors))))

        # Frame 3 must still be Beta_0.jpg, not shifted into frame 2's slot.
        _, ids = index.search(vectors[3:4], 1)
        assert vector_map[int(ids[0][0])]["filename"] == "Beta_0.jpg"

        # The orphaned vector resolves to nothing and must be skipped, never
        # silently substituted with a neighbour.
        assert 2 not in vector_map

    def test_frame_without_a_vector_is_excluded(self, populated_library):
        """A frame row with NULL vector_id is a legal state and must not be
        mapped to some other frame's embedding."""
        import database

        db_path = populated_library["db_path"]
        conn = database.get_db_connection(db_path)
        conn.execute(
            "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
            "VALUES (1, 'Alpha_9999.jpg', '0:09', 9.0, NULL)"
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        conn.close()

        vector_map = load_vector_map(db_path)

        assert total == 7
        assert len(vector_map) == 6
        assert all(m["filename"] != "Alpha_9999.jpg" for m in vector_map.values())

    def test_vector_id_is_unique(self, populated_library):
        """The schema must refuse two frames claiming one embedding."""
        import sqlite3

        import database

        conn = database.get_db_connection(populated_library["db_path"])
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
                "VALUES (1, 'Duplicate.jpg', '0:00', 0.0, 0)"
            )
            conn.commit()
        conn.close()

    def test_desync_is_detectable(self, populated_library):
        """A count mismatch between index and database must be observable --
        this is what the startup integrity check reports."""
        import database

        db_path = populated_library["db_path"]
        vectors = populated_library["vectors"]
        index = build_index(vectors[:4], list(range(4)))

        conn = database.get_db_connection(db_path)
        mapped = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE vector_id IS NOT NULL"
        ).fetchone()[0]
        conn.close()

        assert index.ntotal == 4
        assert mapped == 6
        assert index.ntotal != mapped


class TestNonContiguousIds:
    def test_ids_need_not_be_dense(self, temp_db, unit_vectors):
        """Explicit ids decouple the index from insertion order, which is what
        makes incremental ingest safe. Sparse ids must round-trip."""
        import database

        sparse_ids = [7, 42, 1009, 3]
        vectors = unit_vectors(len(sparse_ids))

        conn = database.get_db_connection(temp_db)
        conn.execute("INSERT INTO videos (title, filename, year) VALUES ('Gamma', 'g.mp4', '2024')")
        for vid, frame_id in enumerate(sparse_ids):
            conn.execute(
                "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
                "VALUES (1, ?, '0:00', 0.0, ?)",
                (f"Gamma_{frame_id}.jpg", frame_id),
            )
        conn.commit()
        conn.close()

        index = build_index(vectors, sparse_ids)
        vector_map = load_vector_map(temp_db)

        for position, frame_id in enumerate(sparse_ids):
            _, ids = index.search(vectors[position : position + 1], 1)
            assert int(ids[0][0]) == frame_id
            assert vector_map[frame_id]["filename"] == f"Gamma_{frame_id}.jpg"
