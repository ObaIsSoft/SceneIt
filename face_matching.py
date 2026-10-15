"""Attribute individual detected faces to named actors.

Pipeline, following the video-face-clustering literature:

    detect -> embed -> cluster -> name the cluster -> attribute every face

Clustering is a means, not the output. The deliverable is one row per
*detected face* in `frame_actors`, carrying the actor, the confidence and the
bounding box. Two faces in one frame get two rows; a face that cannot be named
still gets a row with `actor_id = NULL`, because "a face is here but we don't
know whose" is a real and useful state.

Why cluster at all, when the goal is per-face attribution? Measured on three
real trailers: an individual in-character face matches its actor's red-carpet
portrait at only 0.34-0.62, because costume, makeup and lighting shift the
embedding. A cluster *centroid* reaches 0.70, since averaging cancels that
noise. So the cluster is what gets named reliably, and the name is then
propagated to each of its faces. Faces outside a named cluster fall back to
direct portrait matching at a stricter threshold.
"""

import json

import numpy as np

# Cluster-level naming. Calibrated on five actors whose correct match scored
# 0.42-0.70 with a margin of 0.12-0.45 over the runner-up.
CLUSTER_MATCH_MIN = 0.35
# The gap to the second-best actor. A high score with no margin means the
# portrait matched the population, not the person.
CLUSTER_MATCH_MARGIN = 0.08
# Direct per-face matching skips the averaging that makes clusters reliable,
# so it has to clear a higher bar.
DIRECT_MATCH_MIN = 0.45

CLUSTER_DISTANCE = 0.65


def l2_normalise(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def cluster_faces(embeddings, distance=CLUSTER_DISTANCE):
    """Group face embeddings into identities. Returns an array of labels.

    Average-linkage agglomerative clustering on cosine distance, with no preset
    number of identities -- a cast size is not known in advance.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    embeddings = l2_normalise(embeddings)
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    tree = linkage(pdist(embeddings, metric="cosine"), method="average")
    return fcluster(tree, t=distance, criterion="distance") - 1


def cluster_centroids(embeddings, labels):
    """Mean embedding per cluster, renormalised. {label: centroid}."""
    embeddings = l2_normalise(embeddings)
    centroids = {}
    for label in sorted(set(int(l) for l in labels)):
        members = embeddings[np.asarray(labels) == label]
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroids[label] = centroid / norm
    return centroids


def name_clusters(centroids, reference_embeddings, reference_names,
                  min_similarity=CLUSTER_MATCH_MIN, min_margin=CLUSTER_MATCH_MARGIN):
    """Assign an actor name to each cluster. {label: (name, similarity)}.

    Each actor is claimed by at most one cluster -- the one that matches them
    best. Without that, one strong portrait can win several clusters and the
    same actor ends up attributed to three different people's faces.
    """
    if not centroids or len(reference_embeddings) == 0:
        return {}

    references = l2_normalise(reference_embeddings)
    labels = sorted(centroids)
    matrix = np.vstack([centroids[label] for label in labels])
    scores = matrix @ references.T  # (clusters, actors)

    candidates = []
    for row, label in enumerate(labels):
        order = np.argsort(-scores[row])
        best = int(order[0])
        best_score = float(scores[row][best])
        runner_up = float(scores[row][order[1]]) if len(order) > 1 else -1.0
        if best_score >= min_similarity and (best_score - runner_up) >= min_margin:
            candidates.append((best_score, label, best))

    assigned, taken_actors, taken_clusters = {}, set(), set()
    for score, label, actor_index in sorted(candidates, key=lambda c: -c[0]):
        if actor_index in taken_actors or label in taken_clusters:
            continue
        assigned[label] = (reference_names[actor_index], score)
        taken_actors.add(actor_index)
        taken_clusters.add(label)
    return assigned


def attribute_faces(faces, reference_embeddings=None, reference_names=None,
                    distance=CLUSTER_DISTANCE):
    """Attribute every detected face to an actor where possible.

    `faces` is a list of dicts with at least `embedding`; `frame_id`, `bbox`
    and `det_score` are carried through when present.

    Returns the same list with `cluster_id`, `actor` (or None), `confidence`
    and `method` added to each entry -- one result per face, never per cluster.
    """
    if not faces:
        return []

    embeddings = l2_normalise(np.vstack([f["embedding"] for f in faces]))
    labels = cluster_faces(embeddings, distance=distance)
    centroids = cluster_centroids(embeddings, labels)

    named = {}
    references = None
    if reference_embeddings is not None and len(reference_embeddings) and reference_names:
        references = l2_normalise(reference_embeddings)
        named = name_clusters(centroids, references, reference_names)

    results = []
    for index, face in enumerate(faces):
        label = int(labels[index])
        entry = dict(face)
        entry["cluster_id"] = label
        entry.pop("embedding", None)

        if label in named:
            # Named by the cluster this face belongs to. The confidence stored
            # is this face's OWN similarity to the portrait, not the cluster's
            # -- the row should record evidence about this face.
            actor, _cluster_score = named[label]
            actor_index = reference_names.index(actor)
            entry["actor"] = actor
            entry["confidence"] = round(float(embeddings[index] @ references[actor_index]), 4)
            entry["method"] = "cluster"
        elif references is not None:
            # Unnamed cluster: try this face directly, at a stricter threshold.
            scores = references @ embeddings[index]
            best = int(np.argmax(scores))
            if float(scores[best]) >= DIRECT_MATCH_MIN:
                entry["actor"] = reference_names[best]
                entry["confidence"] = round(float(scores[best]), 4)
                entry["method"] = "direct"
            else:
                entry["actor"] = None
                entry["confidence"] = round(float(scores[best]), 4)
                entry["method"] = "unmatched"
        else:
            entry["actor"] = None
            entry["confidence"] = None
            entry["method"] = "unmatched"

        results.append(entry)
    return results


# --- Persistence ------------------------------------------------------------

def store_reference(conn, name, character=None, image_url=None, embedding=None,
                    tmdb_id=None, profile_path=None, source="tmdb"):
    """Upsert an actor and their reference embedding. Returns the actor id."""
    blob = None
    if embedding is not None:
        blob = l2_normalise(embedding)[0].astype(np.float32).tobytes()

    row = conn.execute("SELECT id FROM actors WHERE name = ?", (name,)).fetchone()
    if row:
        conn.execute(
            "UPDATE actors SET character = COALESCE(?, character), "
            "image = COALESCE(?, image), tmdb_id = COALESCE(?, tmdb_id), "
            "profile_path = COALESCE(?, profile_path), "
            "face_encoding = COALESCE(?, face_encoding), source = ? WHERE id = ?",
            (character, image_url, tmdb_id, profile_path, blob, source, row["id"]),
        )
        return row["id"]

    cur = conn.execute(
        "INSERT INTO actors (name, character, image, tmdb_id, profile_path, "
        "face_encoding, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, character, image_url, tmdb_id, profile_path, blob, source),
    )
    return cur.lastrowid


def load_references(conn):
    """Return (embeddings, names, characters) for actors with a stored face."""
    rows = conn.execute(
        "SELECT name, character, face_encoding FROM actors "
        "WHERE face_encoding IS NOT NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return np.zeros((0, 512), dtype=np.float32), [], []
    embeddings = np.vstack([
        np.frombuffer(r["face_encoding"], dtype=np.float32) for r in rows
    ])
    return embeddings, [r["name"] for r in rows], [r["character"] for r in rows]


def store_attributions(conn, attributions, actor_ids):
    """Write one frame_actors row per detected face. Returns rows written."""
    written = 0
    for face in attributions:
        frame_id = face.get("frame_id")
        if frame_id is None:
            continue
        bbox = face.get("bbox")
        if isinstance(bbox, (list, tuple)):
            bbox = ",".join(str(int(v)) for v in bbox)
        conn.execute(
            "INSERT INTO frame_actors (frame_id, actor_id, cluster_id, confidence, "
            "method, bbox, det_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (frame_id, actor_ids.get(face.get("actor")), face.get("cluster_id"),
             face.get("confidence"), face.get("method"), bbox,
             face.get("det_score")),
        )
        written += 1
    return written


def summarise(attributions):
    """Counts for reporting."""
    named = [f for f in attributions if f.get("actor")]
    by_actor = {}
    for face in named:
        by_actor.setdefault(face["actor"], 0)
        by_actor[face["actor"]] += 1
    return {
        "faces": len(attributions),
        "named": len(named),
        "unnamed": len(attributions) - len(named),
        "identities": len({f["cluster_id"] for f in attributions}),
        "by_actor": dict(sorted(by_actor.items(), key=lambda kv: -kv[1])),
        "by_method": {
            m: sum(1 for f in attributions if f.get("method") == m)
            for m in ("cluster", "direct", "unmatched")
        },
    }
