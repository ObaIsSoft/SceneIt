"""Captioning: availability checks, text tidying, storage and search.

Nothing here loads a 2 GB model or starts a server. The parts worth testing
are the ones that decide whether captioning runs at all and what it does with
the result -- the model itself is exercised by running it, not by unit tests.
"""

import pytest

import captioning

pytestmark = pytest.mark.unit


class TestAvailability:
    """Missing weights must degrade cleanly: ingestion continues without
    captions rather than failing."""

    def test_unavailable_without_weights(self, tmp_path):
        cap = captioning.Captioner(
            model_path=str(tmp_path / "nope.gguf"),
            mmproj_path=str(tmp_path / "nope-mmproj.gguf"),
            server_bin=str(tmp_path / "nope-binary"),
        )
        assert cap.available() is False

    def test_explanation_names_each_missing_piece(self, tmp_path):
        cap = captioning.Captioner(
            model_path=str(tmp_path / "missing-weights.gguf"),
            mmproj_path=str(tmp_path / "missing-proj.gguf"),
            server_bin=str(tmp_path / "missing-binary"),
        )
        why = cap.explain_unavailable()
        assert "missing-binary" in why
        assert "missing-weights.gguf" in why
        assert "missing-proj.gguf" in why

    def test_missing_projector_alone_is_reported(self, tmp_path):
        """Without the projector the model is text-only, and captioning would
        silently describe nothing rather than fail."""
        weights = tmp_path / "w.gguf"
        weights.write_bytes(b"\x00")
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        cap = captioning.Captioner(
            model_path=str(weights),
            mmproj_path=str(tmp_path / "absent-mmproj.gguf"),
            server_bin=str(binary),
        )
        assert cap.available() is False
        assert "projector" in cap.explain_unavailable()

    def test_start_raises_when_unavailable(self, tmp_path):
        cap = captioning.Captioner(
            model_path=str(tmp_path / "a.gguf"),
            mmproj_path=str(tmp_path / "b.gguf"),
            server_bin=str(tmp_path / "c"),
        )
        with pytest.raises(captioning.CaptionerUnavailable):
            cap.start()

    def test_stop_is_safe_when_never_started(self):
        captioning.Captioner().stop()  # must not raise

    def test_caption_of_missing_image_returns_none(self, tmp_path, monkeypatch):
        cap = captioning.Captioner()
        monkeypatch.setattr(cap, "_process", object())  # pretend it is running
        assert cap.caption_image(str(tmp_path / "no-such-frame.jpg")) is None


class TestTidy:
    @pytest.mark.parametrize("raw,expected_start", [
        ("This image shows a man riding a horse.", "A man riding"),
        ("The image shows two people talking.", "Two people"),
        ("In this frame, a woman runs.", "A woman runs"),
        ("The scene shows an explosion.", "An explosion"),
    ])
    def test_preamble_is_stripped(self, raw, expected_start):
        assert captioning._tidy(raw).startswith(expected_start)

    def test_whitespace_is_collapsed(self):
        assert captioning._tidy("  a   man\n\nrides  ") == "A man rides"

    def test_already_clean_text_is_untouched(self):
        assert captioning._tidy("A man rides a horse.") == "A man rides a horse."

    def test_empty_becomes_none(self):
        assert captioning._tidy("") is None
        assert captioning._tidy("   ") is None
        assert captioning._tidy(None) is None


class TestStorage:
    def _frames(self, conn):
        conn.execute("INSERT INTO videos (title, filename) VALUES ('Film', 'f.mp4')")
        for i, name in enumerate(["Film_0.jpg", "Film_1000.jpg", "Film_2000.jpg"]):
            conn.execute(
                "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
                "VALUES (1, ?, '0:00', ?, ?)", (name, float(i), i))
        conn.commit()

    def test_captions_are_written_to_their_frames(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._frames(conn)

        written = captioning.store_captions(conn, {
            "Film_0.jpg": "A man rides a horse across a dune.",
            "Film_1000.jpg": "Two people argue in a doorway.",
        })
        conn.commit()

        rows = conn.execute(
            "SELECT filename, caption, caption_model FROM frames ORDER BY id").fetchall()
        conn.close()

        assert written == 2
        assert rows[0]["caption"].startswith("A man rides")
        assert rows[0]["caption_model"] == captioning.MODEL_LABEL
        assert rows[2]["caption"] is None

    def test_full_paths_are_accepted(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._frames(conn)
        written = captioning.store_captions(
            conn, {"/some/dir/output_frames/Film_0.jpg": "A desert."})
        conn.commit()
        conn.close()
        assert written == 1

    def test_empty_captions_are_skipped(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._frames(conn)
        assert captioning.store_captions(conn, {"Film_0.jpg": None, "Film_1000.jpg": ""}) == 0
        conn.close()


class TestSearch:
    def _captioned(self, conn):
        conn.execute("INSERT INTO videos (title, filename) VALUES ('Dune', 'd.mp4')")
        conn.execute("INSERT INTO videos (title, filename) VALUES ('Oppenheimer', 'o.mp4')")
        rows = [
            (1, "Dune_0.jpg", 0, "A man rides a sandworm across the desert."),
            (1, "Dune_1.jpg", 1, "Two people walk through a sandstorm."),
            (2, "Opp_0.jpg", 2, "A man in a hat smokes a cigarette."),
            (2, "Opp_1.jpg", 3, "A huge explosion lights up the sky."),
        ]
        for video_id, name, vid, caption in rows:
            conn.execute(
                "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id, "
                "caption, caption_model) VALUES (?, ?, '0:00', 0.0, ?, ?, 'test')",
                (video_id, name, vid, caption))
        conn.commit()

    def test_action_query_finds_the_right_frame(self, temp_db):
        """The capability CLIP lacks: 'smoking' should return the smoking
        frame, not merely a frame from a film where smoking occurs."""
        import database

        conn = database.get_db_connection(temp_db)
        self._captioned(conn)
        hits = captioning.search_captions(conn, "smokes cigarette")
        conn.close()

        assert len(hits) == 1
        assert hits[0]["filename"] == "Opp_0.jpg"
        assert hits[0]["video_title"] == "Oppenheimer"

    def test_all_terms_must_match(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._captioned(conn)
        assert captioning.search_captions(conn, "rides sandworm") != []
        assert captioning.search_captions(conn, "rides explosion") == []
        conn.close()

    def test_short_tokens_are_ignored(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._captioned(conn)
        assert captioning.search_captions(conn, "a in的") == []
        conn.close()

    def test_frames_without_captions_never_match(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._captioned(conn)
        conn.execute(
            "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
            "VALUES (1, 'Nocap.jpg', '0:00', 9.0, 99)")
        conn.commit()
        assert all(h["filename"] != "Nocap.jpg"
                   for h in captioning.search_captions(conn, "desert man"))
        conn.close()

    def test_empty_query(self, temp_db):
        import database

        conn = database.get_db_connection(temp_db)
        self._captioned(conn)
        assert captioning.search_captions(conn, "") == []
        conn.close()
