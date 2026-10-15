"""Settings, read from the environment.

This module holds *no data*. It used to carry the entire video library as a
163-line Python literal -- 136 of those lines were hardcoded demo content,
including per-title branches keyed on `if "Dune" in title`. It also read
sceneit_trailers.json, mutated a module-level list and printed to stdout at
import time, which made it impossible to import cheaply or to test.

The library now lives in SQLite and is managed through library.py. Nothing in
this codebase names a specific film.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- Paths ------------------------------------------------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_frames")
INDEX_FILE = os.path.join(OUTPUT_DIR, "frames.npz")
TEMP_VIDEO_DIR = os.getenv("TEMP_VIDEO_DIR", "temp_videos")
# Reference portraits fetched during ingest. Previously a committed directory of
# six JPEGs whose absolute paths leaked through /api/config.
ACTOR_IMAGE_DIR = os.getenv("ACTOR_IMAGE_DIR", os.path.join("media", "actors"))
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback_log.json")

# --- Model ------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 512)

# --- Retrieval --------------------------------------------------------------
# Cosine similarity floors. Still empirically soft: under CLIP two unrelated
# solid colours score ~0.87, so these want calibrating against real queries
# rather than trusted as principled values.
MATCH_THRESHOLD = _env_float("MATCH_THRESHOLD", 0.75)
MATCH_THRESHOLD_MULTIMODAL = _env_float("MATCH_THRESHOLD_MULTIMODAL", 0.68)
# Weighting when a text query refines an image query.
TEXT_FUSION_WEIGHT = _env_float("TEXT_FUSION_WEIGHT", 0.3)
MAX_RESULTS = _env_int("MAX_RESULTS", 12)

# --- Frame sampling ---------------------------------------------------------
FRAMES_TO_SKIP_SECONDS = _env_float("FRAMES_TO_SKIP_SECONDS", 1.0)
MIN_BRIGHTNESS = _env_int("MIN_BRIGHTNESS", 20)
MAX_BRIGHTNESS = _env_int("MAX_BRIGHTNESS", 235)

# --- Captions ---------------------------------------------------------------
# Labels for zero-shot scene classification. Configurable because they are
# content, not logic. The previous implementation also carried per-title
# character substitution tables ("a woman" -> "Sintel"), which only worked for
# the handful of films that happened to be hardcoded.
SCENE_LABELS = [
    label.strip()
    for label in os.getenv(
        "SCENE_LABELS",
        "outdoor scene,indoor scene,close-up shot,wide shot,action sequence,"
        "dramatic scene,conversation,dark scene,bright scene,cityscape,nature scene",
    ).split(",")
    if label.strip()
]
SCENE_LABEL_MIN_CONFIDENCE = _env_float("SCENE_LABEL_MIN_CONFIDENCE", 0.3)

# --- External services ------------------------------------------------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_ENABLED = bool(TMDB_API_KEY) and _env_flag("TMDB_ENABLED", True)

# --- Faces ------------------------------------------------------------------
FACE_RECOGNITION_ENABLED = _env_flag("FACE_RECOGNITION_ENABLED", True)
FACE_MATCH_THRESHOLD = _env_float("FACE_MATCH_THRESHOLD", 0.6)


def __getattr__(name):
    """Resolve DEVICE lazily so importing config does not import torch.

    config is imported by the CLI, by library.py and by the test suite, none of
    which need a tensor library. Eagerly calling torch.cuda.is_available() at
    import time cost seconds on every invocation.
    """
    if name == "DEVICE":
        override = os.getenv("DEVICE")
        if override:
            device = override
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"
        globals()["DEVICE"] = device
        return device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
