"""Endpoint behaviour, including the two that used to return 500s."""

import io
import json

import pytest
from PIL import Image

pytestmark = pytest.mark.api


def jpeg_bytes(colour=(200, 40, 40), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestIndex:
    def test_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"SceneIt" in r.data


class TestConfig:
    def test_returns_library(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        assert isinstance(r.get_json()["video_objects"], list)


class TestSearchValidation:
    def test_missing_file_is_a_400_not_a_crash(self, client):
        r = client.post("/api/search")
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_non_image_upload_is_rejected(self, client):
        r = client.post(
            "/api/search",
            data={"query_image": (io.BytesIO(b"this is not an image"), "evil.jpg")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        assert "error" in r.get_json()


class TestCaption:
    def test_missing_filename_is_a_400(self, client):
        r = client.get("/api/caption")
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_missing_frame_degrades_gracefully(self, client):
        r = client.get("/api/caption?filename=does-not-exist.jpg")
        assert r.status_code == 200
        assert r.get_json()["description"] == "Frame not found."


class TestAllFramesPagination:
    """`page` and `limit` went through a bare int(), so any non-numeric value
    raised ValueError and returned a 500."""

    @pytest.mark.parametrize("qs", ["page=abc", "limit=xyz", "page=&limit=", "page=-5", "limit=0"])
    def test_malformed_pagination_does_not_500(self, client, qs):
        r = client.get(f"/api/all_frames?video_title=all&{qs}")
        assert r.status_code == 200
        body = r.get_json()
        assert {"frames", "total", "has_more"} <= body.keys()

    def test_limit_is_capped(self, client):
        r = client.get("/api/all_frames?video_title=all&limit=100000")
        assert r.status_code == 200
        assert len(r.get_json()["frames"]) <= 100

    def test_shape_is_stable_for_unknown_title(self, client):
        r = client.get("/api/all_frames?video_title=NoSuchFilm")
        assert r.status_code == 200
        body = r.get_json()
        assert body["frames"] == []
        assert body["total"] == 0
        assert body["has_more"] is False


class TestExplore:
    def test_works_with_no_feedback_log(self, client):
        """`seen` was bound inside the 'feedback file exists' branch but read
        unconditionally below it, so a fresh install got a NameError and a 500.
        The client fixture points FEEDBACK_FILE at a path that does not exist.
        """
        r = client.get("/api/explore")
        assert r.status_code == 200
        body = r.get_json()
        assert body["history"] == []
        assert isinstance(body["random"], list)

    def test_works_with_a_malformed_feedback_log(self, client, monkeypatch, tmp_path):
        import app as app_module

        bad = tmp_path / "corrupt.json"
        bad.write_text("{not valid json")
        monkeypatch.setattr(app_module.config, "FEEDBACK_FILE", str(bad))

        r = client.get("/api/explore")
        assert r.status_code == 200
        assert r.get_json()["history"] == []


class TestFeedback:
    def test_records_into_the_temp_file_only(self, client, monkeypatch, tmp_path):
        import app as app_module

        feedback_file = tmp_path / "fb.json"
        monkeypatch.setattr(app_module.config, "FEEDBACK_FILE", str(feedback_file))

        r = client.post(
            "/api/feedback",
            data=json.dumps({"selected_filename": "Alpha_0.jpg", "action": "add"}),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert r.get_json()["status"] == "success"

        written = json.loads(feedback_file.read_text())
        assert written[-1]["selected_filename"] == "Alpha_0.jpg"

    def test_add_then_remove_nets_to_zero(self, client, monkeypatch, tmp_path):
        import app as app_module

        feedback_file = tmp_path / "fb2.json"
        monkeypatch.setattr(app_module.config, "FEEDBACK_FILE", str(feedback_file))

        for action in ("add", "remove"):
            client.post(
                "/api/feedback",
                data=json.dumps({"selected_filename": "Beta_0.jpg", "action": action}),
                content_type="application/json",
            )

        assert app_module.engine.feedback_db.get("Beta_0.jpg", 0) == 0


class TestEngineState:
    def test_feedback_db_exists_before_any_loader_runs(self):
        """/api/search reads engine.feedback_db. It was only created inside
        load_feedback_database(), so any path that skipped that loader raised
        AttributeError mid-request."""
        from app import SearchEngine

        engine = SearchEngine()
        assert engine.feedback_db == {}
        assert engine.vector_map == {}
        assert engine.metadata_db == []
