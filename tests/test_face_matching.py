"""Per-face actor attribution.

The deliverable is one row per *detected face*, not per cluster. Clustering is
a means: a single in-character face matches a red-carpet portrait at only
0.34-0.62, while a cluster centroid reaches 0.70, because averaging cancels the
costume and lighting noise. So the cluster gets named, and the name is
propagated to each of its faces.
"""

import numpy as np
import pytest

import face_matching as fm

pytestmark = pytest.mark.unit

DIM = 512


def identity(seed, n, spread=0.35):
    """n embeddings of one 'person': a base vector plus per-face noise.

    The noise is unit-normalised before scaling, so `spread` means the same
    thing at any dimensionality: cos(base, face) ~= 1/sqrt(1 + spread^2).
    A raw `spread * standard_normal(512)` has norm ~22*spread, which swamps the
    unit base entirely and produces 'faces' with no shared identity at all.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(DIM).astype("float32")
    base /= np.linalg.norm(base)

    noise = rng.standard_normal((n, DIM)).astype("float32")
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    return fm.l2_normalise(base + spread * noise), base


class TestNormalise:
    def test_unit_length(self):
        v = fm.l2_normalise(np.random.default_rng(0).standard_normal((4, DIM)))
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_zero_vector_survives(self):
        v = fm.l2_normalise(np.zeros((1, DIM), dtype="float32"))
        assert not np.isnan(v).any()

    def test_one_dimensional_input(self):
        assert fm.l2_normalise(np.ones(DIM, dtype="float32")).shape == (1, DIM)


class TestClustering:
    def test_separates_distinct_people(self):
        a, _ = identity(1, 6)
        b, _ = identity(2, 5)
        labels = fm.cluster_faces(np.vstack([a, b]))
        # Each group internally consistent, and the two groups distinct.
        assert len(set(labels[:6])) == 1
        assert len(set(labels[6:])) == 1
        assert labels[0] != labels[6]

    def test_handles_empty_and_single(self):
        assert len(fm.cluster_faces(np.zeros((0, DIM)))) == 0
        assert list(fm.cluster_faces(np.ones((1, DIM), dtype="float32"))) == [0]

    def test_centroid_is_unit_length(self):
        a, _ = identity(3, 7)
        cents = fm.cluster_centroids(a, np.zeros(7, dtype=int))
        assert np.isclose(np.linalg.norm(cents[0]), 1.0, atol=1e-5)

    def test_centroid_beats_individual_faces(self):
        """The premise of the whole design: averaging cancels noise, so the
        centroid is closer to the clean reference than any single face."""
        faces, base = identity(4, 12, spread=0.6)
        centroid = fm.cluster_centroids(faces, np.zeros(12, dtype=int))[0]
        per_face = faces @ base
        assert float(centroid @ base) > float(per_face.max())


class TestNaming:
    def test_names_the_right_cluster(self):
        a, base_a = identity(5, 8)
        b, base_b = identity(6, 8)
        faces = np.vstack([a, b])
        labels = fm.cluster_faces(faces)
        cents = fm.cluster_centroids(faces, labels)

        named = fm.name_clusters(cents, np.vstack([base_a, base_b]), ["A", "B"])

        assert named[int(labels[0])][0] == "A"
        assert named[int(labels[8])][0] == "B"

    def test_one_actor_cannot_claim_two_clusters(self):
        """Without this, a single strong portrait wins several clusters and the
        same actor is attributed to several different people's faces."""
        a, base_a = identity(7, 6)
        b, _ = identity(8, 6)
        faces = np.vstack([a, b])
        labels = fm.cluster_faces(faces)
        cents = fm.cluster_centroids(faces, labels)

        named = fm.name_clusters(cents, base_a.reshape(1, -1), ["OnlyActor"])

        assert list(named.values()).count(("OnlyActor", named[list(named)[0]][1])) <= 1
        assert len(named) <= 1

    def test_weak_match_is_refused(self):
        a, _ = identity(9, 5)
        labels = fm.cluster_faces(a)
        cents = fm.cluster_centroids(a, labels)
        unrelated = fm.l2_normalise(
            np.random.default_rng(99).standard_normal((1, DIM)))
        assert fm.name_clusters(cents, unrelated, ["Stranger"]) == {}

    def test_no_references_names_nothing(self):
        a, _ = identity(10, 4)
        cents = fm.cluster_centroids(a, np.zeros(4, dtype=int))
        assert fm.name_clusters(cents, np.zeros((0, DIM)), []) == {}


class TestAttribution:
    def test_every_face_gets_a_row(self):
        """Not one row per cluster -- one per detected face."""
        a, base_a = identity(11, 9)
        b, _ = identity(12, 7)
        faces = [{"embedding": e, "frame_id": i} for i, e in enumerate(np.vstack([a, b]))]

        out = fm.attribute_faces(faces, base_a.reshape(1, -1), ["Actor A"])

        assert len(out) == 16
        assert all("cluster_id" in f and "method" in f for f in out)
        assert sum(1 for f in out if f["actor"] == "Actor A") == 9

    def test_unidentified_faces_are_kept_not_dropped(self):
        """'A face is here but we do not know whose' is a real state."""
        a, base_a = identity(13, 5)
        b, _ = identity(14, 6)
        faces = [{"embedding": e, "frame_id": i} for i, e in enumerate(np.vstack([a, b]))]

        out = fm.attribute_faces(faces, base_a.reshape(1, -1), ["Actor A"])
        unnamed = [f for f in out if f["actor"] is None]

        assert len(unnamed) == 6
        assert all(f["cluster_id"] is not None for f in unnamed)

    def test_confidence_is_per_face_not_per_cluster(self):
        """The stored confidence must describe THIS face, so two faces in one
        cluster should not carry an identical score."""
        a, base_a = identity(15, 10, spread=0.5)
        faces = [{"embedding": e, "frame_id": i} for i, e in enumerate(a)]
        out = fm.attribute_faces(faces, base_a.reshape(1, -1), ["Actor A"])
        named = [f["confidence"] for f in out if f["actor"]]
        assert len(set(named)) > 1

    def test_bbox_and_det_score_survive(self):
        a, base_a = identity(16, 4)
        faces = [{"embedding": e, "frame_id": i, "bbox": (1, 2, 3, 4), "det_score": 0.9}
                 for i, e in enumerate(a)]
        out = fm.attribute_faces(faces, base_a.reshape(1, -1), ["A"])
        assert all(f["bbox"] == (1, 2, 3, 4) and f["det_score"] == 0.9 for f in out)

    def test_embedding_is_not_leaked_into_the_row(self):
        a, _ = identity(17, 3)
        out = fm.attribute_faces([{"embedding": e, "frame_id": i} for i, e in enumerate(a)])
        assert all("embedding" not in f for f in out)

    def test_no_references_still_clusters(self):
        a, _ = identity(18, 5)
        out = fm.attribute_faces([{"embedding": e, "frame_id": i} for i, e in enumerate(a)])
        assert all(f["actor"] is None and f["method"] == "unmatched" for f in out)
        assert len({f["cluster_id"] for f in out}) == 1

    def test_empty_input(self):
        assert fm.attribute_faces([]) == []


class TestPersistence:
    def test_reference_round_trip(self, temp_db):
        import database

        _, base = identity(20, 1)
        conn = database.get_db_connection(temp_db)
        actor_id = fm.store_reference(
            conn, "Someone", character="Lead",
            image_url="/api/actor_image/x.jpg", embedding=base,
            tmdb_id=123, profile_path="/x.jpg")
        conn.commit()

        embeddings, names, characters = fm.load_references(conn)
        conn.close()

        assert actor_id and names == ["Someone"] and characters == ["Lead"]
        assert embeddings.shape == (1, DIM)
        assert float(fm.l2_normalise(base)[0] @ embeddings[0]) > 0.999

    def test_reference_upsert_does_not_duplicate(self, temp_db):
        import database

        _, base = identity(21, 1)
        conn = database.get_db_connection(temp_db)
        first = fm.store_reference(conn, "Dup", embedding=base)
        second = fm.store_reference(conn, "Dup", character="Updated")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM actors").fetchone()[0]
        row = conn.execute("SELECT character FROM actors WHERE id = ?", (first,)).fetchone()
        conn.close()

        assert first == second and count == 1
        assert row["character"] == "Updated"

    def test_two_faces_in_one_frame_are_both_stored(self, temp_db):
        """The old schema keyed (frame_id, actor_id), so it could not represent
        two detections in a single frame."""
        import database

        conn = database.get_db_connection(temp_db)
        conn.execute("INSERT INTO videos (title, filename) VALUES ('V', 'v.mp4')")
        conn.execute("INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
                     "VALUES (1, 'V_0.jpg', '0:00', 0.0, 0)")
        a_id = fm.store_reference(conn, "A", embedding=identity(22, 1)[1])
        conn.commit()

        written = fm.store_attributions(conn, [
            {"frame_id": 1, "actor": "A", "cluster_id": 0, "confidence": 0.7,
             "method": "cluster", "bbox": (0, 0, 10, 10), "det_score": 0.9},
            {"frame_id": 1, "actor": "A", "cluster_id": 0, "confidence": 0.6,
             "method": "cluster", "bbox": (20, 20, 30, 30), "det_score": 0.8},
            {"frame_id": 1, "actor": None, "cluster_id": 5, "confidence": 0.1,
             "method": "unmatched", "bbox": (40, 40, 50, 50), "det_score": 0.7},
        ], {"A": a_id})
        conn.commit()

        rows = conn.execute("SELECT * FROM frame_actors ORDER BY id").fetchall()
        conn.close()

        assert written == 3 and len(rows) == 3
        assert rows[0]["bbox"] == "0,0,10,10"
        assert rows[2]["actor_id"] is None   # unidentified face is still recorded


class TestSummary:
    def test_counts(self):
        a, base_a = identity(23, 6)
        b, _ = identity(24, 4)
        faces = [{"embedding": e, "frame_id": i} for i, e in enumerate(np.vstack([a, b]))]
        s = fm.summarise(fm.attribute_faces(faces, base_a.reshape(1, -1), ["A"]))

        assert s["faces"] == 10
        assert s["named"] == 6
        assert s["unnamed"] == 4
        assert s["by_actor"]["A"] == 6
