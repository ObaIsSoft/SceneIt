"""Audit face detection and clustering across the indexed frames.

Answers the question the roadmap depends on: can cast be discovered from the
footage itself, instead of from hand-placed reference portraits?

The method follows the video-face-clustering literature -- detect, embed,
aggregate, then cluster with an unknown number of identities:

  1. detect every face in every indexed frame (buffalo_l / RetinaFace)
  2. embed each with ArcFace (w600k_r50), 512-d, L2-normalised
  3. agglomerative clustering on cosine distance, no target cluster count
  4. report cluster sizes and write a montage per cluster for eyeballing

Run:  python scripts/face_audit.py [--min-size 40] [--distance 0.65]
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def collect_faces(frame_paths, min_size):
    from face_id import FaceDetector

    detector = FaceDetector()
    if detector.app is None:
        print("Face model unavailable. Install with: pip install -e '.[faces]'")
        return []

    records = []
    for i, path in enumerate(frame_paths, 1):
        image = cv2.imread(path)
        if image is None:
            continue
        for face in detector.app.get(image):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            w, h = x2 - x1, y2 - y1
            # Small faces carry unreliable embeddings; the literature filters
            # them rather than letting them pollute the clustering.
            if min(w, h) < min_size:
                continue
            emb = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            records.append({
                "path": path,
                "film": os.path.basename(path).rsplit("_", 1)[0],
                "bbox": (max(0, x1), max(0, y1), x2, y2),
                "size": min(w, h),
                "det_score": float(getattr(face, "det_score", 0.0)),
                "embedding": emb / norm,
            })
        if i % 50 == 0:
            print(f"  scanned {i}/{len(frame_paths)} frames, {len(records)} faces so far")
    return records


def cluster(records, distance):
    """Agglomerative clustering with no preset number of identities."""
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("scikit-learn unavailable; skipping clustering.")
        return None

    if len(records) < 2:
        return None

    matrix = np.vstack([r["embedding"] for r in records])
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(matrix)


def write_montage(records, labels, out_dir, per_cluster=8):
    os.makedirs(out_dir, exist_ok=True)
    by_label = defaultdict(list)
    for record, label in zip(records, labels):
        by_label[int(label)].append(record)

    written = []
    for label, members in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        members = sorted(members, key=lambda r: -r["size"])[:per_cluster]
        tiles = []
        for member in members:
            x1, y1, x2, y2 = member["bbox"]
            image = cv2.imread(member["path"])
            if image is None:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tiles.append(cv2.resize(crop, (112, 112)))
        if not tiles:
            continue
        path = os.path.join(out_dir, f"cluster_{label:02d}_n{len(by_label[label])}.jpg")
        cv2.imwrite(path, np.hstack(tiles))
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-size", type=int, default=40,
                        help="ignore faces smaller than this many pixels")
    parser.add_argument("--distance", type=float, default=0.65,
                        help="cosine distance threshold for clustering")
    parser.add_argument("--out", default=".local-data/face_clusters")
    args = parser.parse_args()

    frames = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*.jpg")))
    if not frames:
        print(f"No frames in {config.OUTPUT_DIR}. Run: python app.py --extract")
        return 1

    print(f"Scanning {len(frames)} frames for faces (min size {args.min_size}px)...\n")
    records = collect_faces(frames, args.min_size)
    if not records:
        print("No faces detected.")
        return 0

    frames_with_faces = len({r["path"] for r in records})
    print(f"\n{'=' * 62}")
    print(f"  faces detected      : {len(records)}")
    print(f"  frames with a face  : {frames_with_faces}/{len(frames)} "
          f"({frames_with_faces / len(frames) * 100:.1f}%)")

    per_film = Counter(r["film"] for r in records)
    print("\n  faces per film:")
    for film, count in per_film.most_common():
        film_frames = len([f for f in frames if os.path.basename(f).startswith(film + "_")])
        print(f"    {film[:34]:<34} {count:>4} faces across {film_frames} frames")

    sizes = np.array([r["size"] for r in records])
    print(f"\n  face size px: median {np.median(sizes):.0f}, "
          f"min {sizes.min()}, max {sizes.max()}")

    labels = cluster(records, args.distance)
    if labels is None:
        return 0

    counts = Counter(labels.tolist())
    singletons = sum(1 for v in counts.values() if v == 1)
    print(f"\n  identities found    : {len(counts)} "
          f"({singletons} singletons at distance {args.distance})")
    print("\n  largest clusters:")
    for label, n in counts.most_common(10):
        films = Counter(r["film"] for r, lab in zip(records, labels) if lab == label)
        spread = ", ".join(f"{k[:20]}:{v}" for k, v in films.most_common(3))
        print(f"    cluster {label:>3}  {n:>3} faces   [{spread}]")

    written = write_montage(records, labels, args.out)
    print(f"\n  wrote {len(written)} montages to {args.out}/")
    print(f"{'=' * 62}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
