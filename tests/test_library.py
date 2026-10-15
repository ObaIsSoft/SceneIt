"""The library as data.

Covers what replaced config.VIDEOS_TO_PROCESS: source classification, title and
filename derivation, add/remove, ingest status, and the API serialisation that
must never leak a filesystem path.
"""

import pytest

import library

pytestmark = pytest.mark.unit


class TestClassifySource:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://m.youtube.com/watch?v=abc123",
        "https://www.youtube.com/shorts/abc123",
    ])
    def test_youtube_variants(self, url):
        assert library.classify_source(url) == "youtube"

    def test_plain_http(self):
        assert library.classify_source("https://example.com/film.mp4") == "http"

    @pytest.mark.parametrize("path", ["/tmp/film.mp4", "film.mp4", "", None])
    def test_local(self, path):
        assert library.classify_source(path) == "local"


class TestYoutubeId:
    """The old extraction was `url.split('v=')[1].split('&')[0]`, which raised
    IndexError on every URL form except watch?v=."""

    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/watch?v=Way9Dexny3w", "Way9Dexny3w"),
        ("https://www.youtube.com/watch?v=Way9Dexny3w&t=10s", "Way9Dexny3w"),
        ("https://youtu.be/Way9Dexny3w", "Way9Dexny3w"),
        ("https://www.youtube.com/shorts/Way9Dexny3w", "Way9Dexny3w"),
        ("https://www.youtube.com/embed/Way9Dexny3w", "Way9Dexny3w"),
        ("https://example.com/film.mp4", None),
        ("https://www.youtube.com/feed/subscriptions", None),
    ])
    def test_extraction(self, url, expected):
        assert library.youtube_id(url) == expected


class TestSafeFilename:
    def test_spaces_and_punctuation_are_stripped(self):
        name = library.safe_filename("Dune: Part Two!")
        assert "/" not in name and ":" not in name
        assert name.endswith(".mp4")

    def test_path_traversal_cannot_escape(self):
        """A title is attacker-influenced when discovery pulls it from an API."""
        for hostile in ("../../etc/passwd", "../../../x", "/absolute/evil"):
            name = library.safe_filename(hostile)
            assert ".." not in name
            assert not name.startswith("/")

    def test_extension_comes_from_the_url(self):
        assert library.safe_filename("X", "https://e.com/clip.webm").endswith(".webm")

    def test_empty_title_still_yields_a_name(self):
        assert library.safe_filename("").endswith(".mp4")


class TestDeriveTitle:
    def test_from_a_url_path(self):
        assert library.derive_title("https://e.com/sintel_trailer-720p.mp4")

    def test_youtube_without_a_path_uses_the_id(self):
        title = library.derive_title("https://www.youtube.com/watch?v=abc123")
        assert "abc123" in title


class TestAddAndList:
    def test_add_then_list(self, temp_db):
        row, created = library.add_source(
            "https://e.com/a.mp4", title="Film A", year="2024", db_path=temp_db)
        assert created
        assert row["title"] == "Film A"
        assert row["source_type"] == "http"
        assert row["status"] == library.STATUS_PENDING

        sources = library.list_sources(db_path=temp_db)
        assert len(sources) == 1
        assert sources[0]["frame_count"] == 0

    def test_duplicate_title_is_not_created_twice(self, temp_db):
        library.add_source("https://e.com/a.mp4", title="Film A", db_path=temp_db)
        row, created = library.add_source(
            "https://e.com/other.mp4", title="Film A", db_path=temp_db)
        assert created is False
        assert len(library.list_sources(db_path=temp_db)) == 1

    def test_replace_overwrites(self, temp_db):
        library.add_source("https://e.com/a.mp4", title="Film A", db_path=temp_db)
        row, _ = library.add_source(
            "https://e.com/b.mp4", title="Film A", replace=True, db_path=temp_db)
        assert row["source_url"] == "https://e.com/b.mp4"
        assert len(library.list_sources(db_path=temp_db)) == 1

    def test_missing_local_file_is_refused(self, temp_db):
        with pytest.raises(FileNotFoundError):
            library.add_source("/no/such/file.mp4", title="Ghost", db_path=temp_db)

    def test_local_file_is_accepted(self, temp_db, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00\x00")
        row, created = library.add_source(str(f), title="Local Clip", db_path=temp_db)
        assert created and row["source_type"] == "local"

    def test_title_is_required_somehow(self, temp_db):
        with pytest.raises((ValueError, FileNotFoundError)):
            library.add_source("", db_path=temp_db)


class TestStatusTransitions:
    def test_mark_ingested(self, temp_db):
        row, _ = library.add_source("https://e.com/a.mp4", title="A", db_path=temp_db)
        library.mark_ingested(row["id"], duration_sec=120.5, db_path=temp_db)
        updated = library.get_source(row["id"], db_path=temp_db)
        assert updated["status"] == library.STATUS_INGESTED
        assert updated["duration_sec"] == pytest.approx(120.5)
        assert updated["ingested_at"]

    def test_mark_failed_records_the_reason(self, temp_db):
        row, _ = library.add_source("https://e.com/a.mp4", title="A", db_path=temp_db)
        library.mark_failed(row["id"], "download returned 403", db_path=temp_db)
        updated = library.get_source(row["id"], db_path=temp_db)
        assert updated["status"] == library.STATUS_FAILED
        assert "403" in updated["error"]

    def test_filter_by_status(self, temp_db):
        a, _ = library.add_source("https://e.com/a.mp4", title="A", db_path=temp_db)
        library.add_source("https://e.com/b.mp4", title="B", db_path=temp_db)
        library.mark_ingested(a["id"], db_path=temp_db)

        assert len(library.list_sources(library.STATUS_INGESTED, db_path=temp_db)) == 1
        assert len(library.list_sources(library.STATUS_PENDING, db_path=temp_db)) == 1


class TestLookup:
    def test_by_id_and_by_title(self, temp_db):
        row, _ = library.add_source("https://e.com/a.mp4", title="Film A", db_path=temp_db)
        assert library.get_source(row["id"], db_path=temp_db)["title"] == "Film A"
        assert library.get_source("Film A", db_path=temp_db)["id"] == row["id"]

    def test_missing_returns_none(self, temp_db):
        assert library.get_source(9999, db_path=temp_db) is None
        assert library.get_source("Nope", db_path=temp_db) is None

    def test_substring_search(self, temp_db):
        library.add_source("https://e.com/a.mp4", title="Dune: Part Two", db_path=temp_db)
        library.add_source("https://e.com/b.mp4", title="Oppenheimer", db_path=temp_db)
        assert len(library.find_sources(["dune"], db_path=temp_db)) == 1
        assert len(library.find_sources(["DUNE"], db_path=temp_db)) == 1
        assert library.find_sources(["nothing"], db_path=temp_db) == []


class TestRemoval:
    def test_remove(self, temp_db):
        row, _ = library.add_source("https://e.com/a.mp4", title="A", db_path=temp_db)
        assert library.remove_source(row["id"], db_path=temp_db) is True
        assert library.list_sources(db_path=temp_db) == []

    def test_remove_missing(self, temp_db):
        assert library.remove_source(123, db_path=temp_db) is False

    def test_frames_cascade(self, temp_db):
        import database

        row, _ = library.add_source("https://e.com/a.mp4", title="A", db_path=temp_db)
        conn = database.get_db_connection(temp_db)
        conn.execute(
            "INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id) "
            "VALUES (?, 'A_0.jpg', '0:00', 0.0, 0)", (row["id"],))
        conn.commit()
        conn.close()

        library.remove_source(row["id"], db_path=temp_db)

        conn = database.get_db_connection(temp_db)
        remaining = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        conn.close()
        assert remaining == 0


class TestApiSerialisation:
    def test_never_leaks_paths_or_source_urls(self, temp_db, tmp_path):
        """/api/config used to return absolute paths like
        /Users/<name>/.../zendaya.jpg, which leaked the server layout and
        rendered as broken images in the browser."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00")
        row, _ = library.add_source(str(f), title="Local", db_path=temp_db)

        payload = library.to_api_dict(library.get_source(row["id"], db_path=temp_db))

        assert "source_url" not in payload
        blob = repr(payload)
        assert str(tmp_path) not in blob
        assert "/Users/" not in blob

    def test_shape_is_complete(self, temp_db):
        row, _ = library.add_source(
            "https://e.com/a.mp4", title="A", year="2024", db_path=temp_db)
        payload = library.to_api_dict(library.get_source(row["id"], db_path=temp_db))
        assert {"id", "title", "year", "status", "actors", "streaming_on"} <= payload.keys()
        assert payload["actors"] == []


class TestMetadataEnrichment:
    def test_merge_does_not_disturb_ingest_state(self, temp_db):
        row, _ = library.add_source("https://e.com/a.mp4", title="A", db_path=temp_db)
        library.mark_ingested(row["id"], db_path=temp_db)

        library.update_metadata(
            row["id"],
            synopsis="A film about things.",
            actors=[{"name": "Someone", "character": "Lead", "image": "/api/actor_image/x.jpg"}],
            db_path=temp_db,
        )
        updated = library.get_source(row["id"], db_path=temp_db)
        assert updated["status"] == library.STATUS_INGESTED
        assert updated["synopsis"] == "A film about things."
        assert updated["actors"][0]["name"] == "Someone"

    def test_update_missing_source(self, temp_db):
        assert library.update_metadata(9999, synopsis="x", db_path=temp_db) is False
