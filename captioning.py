"""Scene captions from Qwen3.5-2B, served by llama.cpp.

Why captions at all
-------------------
CLIP matches *setting*, not *action*. Measured on three real trailers, every
action query landed in an undifferentiated ~0.27 similarity band and chose the
wrong film: "a man smoking a cigarette" returned Blade Runner over Oppenheimer,
"someone firing a gun" returned Dune. A single frame of "riding a horse" and
"standing beside a horse" look nearly identical to a frame encoder, and the
difference between them is temporal.

Captioning turns the action into *words*, which text search handles well. The
cost lands at ingest, once per frame, so it never touches the ~22 ms query path.

Why llama.cpp and not transformers
----------------------------------
Qwen3.5-2B as fp16 safetensors is 4.55 GB and needs torch (~550 MB more) --
which would undo the whole point of moving to ONNX Runtime. The same model as
GGUF Q4_K_M is 1.28 GB, plus a 0.67 GB vision projector, and runs with no
Python ML stack at all.

Why a server and not a CLI per image
------------------------------------
Loading ~2 GB of weights per frame would dominate everything else. The model is
loaded once into a long-lived `llama-server` process and every frame is sent to
it over HTTP; the process is shut down when captioning finishes.
"""

import atexit
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import config

MODEL_LABEL = "Qwen3.5-2B-Q4_K_M"


class CaptionerUnavailable(RuntimeError):
    """Binary or weights missing. Ingestion continues without captions."""


class Captioner:
    """Owns a llama-server process and captions images against it."""

    def __init__(self, model_path=None, mmproj_path=None, port=None,
                 server_bin=None, prompt=None):
        self.model_path = model_path or config.CAPTION_MODEL_PATH
        self.mmproj_path = mmproj_path or config.CAPTION_MMPROJ_PATH
        self.port = port or config.CAPTION_SERVER_PORT
        self.server_bin = server_bin or config.CAPTION_SERVER_BIN
        self.prompt = prompt or config.CAPTION_PROMPT
        self._process = None
        self._base_url = f"http://127.0.0.1:{self.port}"

    # --- Lifecycle --------------------------------------------------------

    def available(self):
        """True when the binary and both model files are present."""
        import shutil

        if shutil.which(self.server_bin) is None and not os.path.exists(self.server_bin):
            return False
        return os.path.exists(self.model_path) and os.path.exists(self.mmproj_path)

    def explain_unavailable(self):
        import shutil

        missing = []
        if shutil.which(self.server_bin) is None and not os.path.exists(self.server_bin):
            missing.append(f"{self.server_bin} (llama.cpp server binary) not on PATH")
        if not os.path.exists(self.model_path):
            missing.append(f"weights missing: {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            missing.append(f"vision projector missing: {self.mmproj_path}")
        return "; ".join(missing) or "unknown"

    def start(self, timeout=180):
        """Launch llama-server and block until it answers."""
        if self._process is not None:
            return
        if not self.available():
            raise CaptionerUnavailable(self.explain_unavailable())

        cmd = [
            self.server_bin,
            "-m", self.model_path,
            "--mmproj", self.mmproj_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-c", str(config.CAPTION_CONTEXT),
            "--no-warmup",
        ]
        if config.CAPTION_THREADS:
            cmd += ["-t", str(config.CAPTION_THREADS)]

        print(f"--- Starting caption server ({MODEL_LABEL}) on port {self.port} ---")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.stop)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                self._process = None
                raise CaptionerUnavailable(
                    f"{self.server_bin} exited during startup "
                    f"(code {self._process.returncode if self._process else '?'})")
            try:
                with urllib.request.urlopen(f"{self._base_url}/health", timeout=2):
                    print("--- Caption server ready ---")
                    return
            except Exception:  # noqa: BLE001 - still booting
                time.sleep(1.5)

        self.stop()
        raise CaptionerUnavailable(f"caption server did not become ready in {timeout}s")

    def stop(self):
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # --- Captioning -------------------------------------------------------

    def caption_image(self, image_path, prompt=None, max_tokens=None, timeout=120):
        """Caption one image. Returns the text, or None on failure."""
        if self._process is None:
            self.start()
        if not os.path.exists(image_path):
            return None

        with open(image_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")

        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    {"type": "text", "text": prompt or self.prompt},
                ],
            }],
            "max_tokens": max_tokens or config.CAPTION_MAX_TOKENS,
            # Deterministic: the same frame should caption the same way on a
            # re-run, so captions can be diffed between model versions.
            "temperature": 0.0,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as e:
            print(f"  caption failed ({os.path.basename(image_path)}): HTTP {e.code}")
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  caption failed ({os.path.basename(image_path)}): {e}")
            return None

        choices = body.get("choices") or []
        if not choices:
            return None
        text = (choices[0].get("message") or {}).get("content") or ""
        return _tidy(text)

    def caption_many(self, image_paths, progress_every=10):
        """Caption a batch against one loaded model. {path: caption|None}."""
        results = {}
        total = len(image_paths)
        for index, path in enumerate(image_paths, 1):
            results[path] = self.caption_image(path)
            if progress_every and index % progress_every == 0:
                done = sum(1 for v in results.values() if v)
                print(f"  captioned {index}/{total} ({done} succeeded)")
        return results


def _tidy(text):
    """Trim the preamble VLMs habitually add."""
    text = " ".join((text or "").split()).strip()
    for prefix in ("This image shows ", "The image shows ", "This frame shows ",
                   "The frame shows ", "In this image, ", "In this frame, ",
                   "This is ", "The scene shows "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    if not text:
        return None
    # Always capitalise: captions are displayed, and a VLM may return lowercase
    # whether or not a preamble was stripped.
    return text[:1].upper() + text[1:]


# --- Persistence ------------------------------------------------------------

def store_captions(conn, captions_by_filename, model_label=MODEL_LABEL):
    """Write captions onto their frame rows. Returns rows updated."""
    updated = 0
    for filename, caption in captions_by_filename.items():
        if not caption:
            continue
        cursor = conn.execute(
            "UPDATE frames SET caption = ?, caption_model = ? WHERE filename = ?",
            (caption, model_label, os.path.basename(filename)),
        )
        updated += cursor.rowcount
    return updated


def search_captions(conn, query, limit=20):
    """Keyword search over stored captions.

    Deliberately simple: this is the text half of a hybrid search, and the
    vector half already handles semantics. FTS5 is the upgrade path if ranking
    quality becomes the bottleneck.
    """
    terms = [t for t in (query or "").lower().split() if len(t) > 2]
    if not terms:
        return []

    clause = " AND ".join("LOWER(f.caption) LIKE ?" for _ in terms)
    rows = conn.execute(
        f"SELECT f.filename, f.timestamp, f.time_sec, f.caption, v.title AS video_title "
        f"FROM frames f JOIN videos v ON f.video_id = v.id "
        f"WHERE f.caption IS NOT NULL AND {clause} "
        f"ORDER BY f.id LIMIT ?",
        [*(f"%{t}%" for t in terms), limit],
    ).fetchall()
    return [dict(r) for r in rows]
