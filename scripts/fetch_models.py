"""Download every model the image bakes in.

Run at Docker build time so a cold start never waits on a model fetch. That is
not hypothetical: the InsightFace CDN served buffalo_l at ~60 KB/s during
development, which is over an hour for one 280 MB pack. Every URL here is
pinned to a specific repo and revision, and sizes are asserted, so a build
fails loudly rather than shipping a truncated model.

    python scripts/fetch_models.py --set serve    # CLIP + faces (~340 MB)
    python scripts/fetch_models.py --set caption  # Qwen3.5-2B GGUF (~1.95 GB)
    python scripts/fetch_models.py --set all
"""

import argparse
import os
import sys
import urllib.request

HF = "https://huggingface.co"

# name -> (url, destination, approximate MB, tolerance fraction)
MODELS = {
    # --- Search: CLIP ViT-B/32, split towers, int8 -----------------------
    "clip-vision": (
        f"{HF}/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_int8.onnx",
        "clip/vision_model_int8.onnx", 88.6, 0.15),
    "clip-text": (
        f"{HF}/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_int8.onnx",
        "clip/text_model_int8.onnx", 64.1, 0.15),

    # --- Faces: detection + recognition only -----------------------------
    # The buffalo_l pack ships five models (~325 MB); this pipeline uses two.
    "face-detect": (
        f"{HF}/public-data/insightface/resolve/main/models/buffalo_l.zip",
        "faces/buffalo_l.zip", 288.6, 0.10),

    # --- Captioning: Qwen3.5-2B, GGUF ------------------------------------
    "caption-weights": (
        f"{HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
        "Qwen3.5-2B-Q4_K_M.gguf", 1280.0, 0.10),
    # Without the projector the model is text-only and captioning silently
    # describes nothing.
    "caption-mmproj": (
        f"{HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf",
        "Qwen3.5-2B-mmproj-F16.gguf", 668.2, 0.10),
}

SETS = {
    "serve": ["clip-vision", "clip-text", "face-detect"],
    "caption": ["caption-weights", "caption-mmproj"],
    "all": list(MODELS),
}


def download(url, dest, expected_mb, tolerance, force=False):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    if os.path.exists(dest) and not force:
        actual = os.path.getsize(dest) / 1e6
        if abs(actual - expected_mb) <= expected_mb * tolerance:
            print(f"  cached   {os.path.basename(dest):<34} {actual:8.1f} MB")
            return True
        print(f"  size mismatch on {dest} ({actual:.1f} MB), refetching")

    print(f"  fetching {os.path.basename(dest):<34} ({expected_mb:.0f} MB expected)")
    request = urllib.request.Request(url, headers={"User-Agent": "SceneIt/0.1"})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  FAILED   {os.path.basename(dest)}: {e}")
        return False

    actual = os.path.getsize(tmp) / 1e6
    if abs(actual - expected_mb) > expected_mb * tolerance:
        os.remove(tmp)
        print(f"  FAILED   {os.path.basename(dest)}: got {actual:.1f} MB, "
              f"expected ~{expected_mb:.0f} MB")
        return False

    os.replace(tmp, dest)
    print(f"  ok       {os.path.basename(dest):<34} {actual:8.1f} MB")
    return True


def unpack_faces(model_dir):
    """Extract buffalo_l and drop the three models this pipeline never loads."""
    import zipfile

    archive = os.path.join(model_dir, "faces", "buffalo_l.zip")
    target = os.path.join(model_dir, "faces", "buffalo_l")
    if not os.path.exists(archive):
        return
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)

    # Keep detection and recognition; discard 3D landmarks (137 MB),
    # 2D landmarks and gender/age, none of which are used.
    keep = {"det_10g.onnx", "w600k_r50.onnx"}
    freed = 0
    for name in os.listdir(target):
        if name.endswith(".onnx") and name not in keep:
            path = os.path.join(target, name)
            freed += os.path.getsize(path)
            os.remove(path)
    os.remove(archive)
    print(f"  pruned unused face models, freed {freed / 1e6:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="all", choices=sorted(SETS))
    parser.add_argument("--dir", default=os.getenv("MODEL_DIR", "models"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Fetching '{args.set}' model set into {args.dir}/\n")
    failures = []
    for name in SETS[args.set]:
        url, rel, mb, tol = MODELS[name]
        if not download(url, os.path.join(args.dir, rel), mb, tol, args.force):
            failures.append(name)

    if args.set in ("serve", "all") and "face-detect" not in failures:
        unpack_faces(args.dir)

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(args.dir) for f in files
    )
    print(f"\n  total on disk: {total / 1e9:.2f} GB")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
