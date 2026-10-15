"""Frame normalisation must be identical on the ingest and query paths.

The bug this guards against: extraction stored frames untouched while
/api/search cropped the uploaded image. Query vectors and index vectors were
therefore computed from differently-shaped inputs, and a frame matched against
its own stored vector at 0.9404 rather than 0.9989 -- enough for a neighbouring
frame to win. Self-retrieval failed on a 29-frame index.

Idempotency is the property that makes the two paths agree: ingest stores
normalize_frame(raw), the query computes normalize_frame(stored), and those are
only equal if normalising twice changes nothing.
"""

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def letterboxed(content=(90, 140, 200), bar_rows=80, h=360, w=640):
    """A bright frame inside black bars, as a 2.35:1 film in a 16:9 container."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[bar_rows : h - bar_rows, :] = content
    return img


def pillarboxed(content=(200, 120, 60), bar_cols=100, h=360, w=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, bar_cols : w - bar_cols] = content
    return img


class TestLetterboxRemoval:
    def test_horizontal_bars_are_removed(self):
        from app import normalize_frame

        out = normalize_frame(letterboxed(bar_rows=80))
        assert out.shape[0] == 360 - 160
        assert out.shape[1] == 640

    def test_vertical_bars_are_removed(self):
        from app import normalize_frame

        out = normalize_frame(pillarboxed(bar_cols=100))
        assert out.shape[1] == 640 - 200
        assert out.shape[0] == 360

    def test_tight_frame_is_untouched(self):
        from app import normalize_frame

        img = np.full((120, 200, 3), 180, dtype=np.uint8)
        assert normalize_frame(img).shape == img.shape


class TestIdempotency:
    """The symmetry guarantee."""

    @pytest.mark.parametrize(
        "image",
        [letterboxed(), pillarboxed(), np.full((100, 100, 3), 200, dtype=np.uint8)],
        ids=["letterboxed", "pillarboxed", "tight"],
    )
    def test_normalising_twice_changes_nothing(self, image):
        from app import normalize_frame

        once = normalize_frame(image)
        twice = normalize_frame(once)
        assert once.shape == twice.shape
        assert np.array_equal(once, twice)

    def test_jpeg_round_trip_stays_stable(self, tmp_path):
        """Ingest writes a JPEG and the query re-reads one, so normalisation has
        to survive lossy re-encoding without shifting the crop box."""
        from app import normalize_frame

        path = tmp_path / "frame.jpg"
        cv2.imwrite(str(path), normalize_frame(letterboxed()))
        reloaded = cv2.imread(str(path))

        assert np.array_equal(normalize_frame(reloaded).shape, reloaded.shape)


class TestDegenerateInput:
    def test_all_black_frame_is_not_cropped_away(self):
        """findNonZero returns None here. Returning an empty array would make
        the embedder fail on a legitimately dark frame."""
        from app import normalize_frame

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = normalize_frame(img)
        assert out is not None
        assert out.size > 0

    def test_nearly_black_frame_keeps_usable_size(self):
        """A single bright pixel must not crop the frame to 1x1."""
        from app import normalize_frame

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[100, 100] = (255, 255, 255)
        out = normalize_frame(img)
        assert out.shape[0] >= 16 and out.shape[1] >= 16

    def test_none_is_handled(self):
        from app import normalize_frame

        assert normalize_frame(None) is None
