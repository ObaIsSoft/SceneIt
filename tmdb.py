"""TMDB metadata: cast lists and reference portraits.

TMDB performs no face recognition. It supplies the *labels* -- who is in a film,
which character they play, and a portrait of them. Recognition is done locally
by ArcFace (see face_id.py). The two meet in face_matching.py, where a portrait
embedding names a cluster of faces discovered in the footage.

This replaces the hand-written actor tables that used to sit in config.py behind
`if "Dune" in title` branches, and the six portrait JPEGs committed to the repo.

Degrades to a no-op without TMDB_API_KEY, so every caller must handle an empty
cast rather than assume enrichment happened.
"""

import json
import os
import urllib.parse
import urllib.request

import config

API_ROOT = "https://api.themoviedb.org/3"
IMAGE_ROOT = "https://image.tmdb.org/t/p"
# w500 is plenty: ArcFace works from a 112x112 aligned crop, so a larger
# download buys nothing but bandwidth.
PORTRAIT_SIZE = "w500"
TIMEOUT = 20


class TMDBUnavailable(RuntimeError):
    """No API key, or the service could not be reached."""


def enabled():
    return bool(config.TMDB_API_KEY)


def _get(path, **params):
    if not enabled():
        raise TMDBUnavailable("TMDB_API_KEY is not set")
    params["api_key"] = config.TMDB_API_KEY
    url = f"{API_ROOT}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise TMDBUnavailable("TMDB rejected the API key") from e
        raise TMDBUnavailable(f"TMDB HTTP {e.code}") from e
    except Exception as e:  # noqa: BLE001 - network layer, surface as one type
        raise TMDBUnavailable(str(e)) from e


def portrait_url(profile_path, size=PORTRAIT_SIZE):
    """Build a full portrait URL from TMDB's relative profile path."""
    if not profile_path:
        return None
    return f"{IMAGE_ROOT}/{size}{profile_path}"


def search_movie(title, year=None):
    """Return the best-matching movie, or None.

    Titles coming from YouTube are messy ("Dune: Part Two | Official Trailer"),
    so the noisiest suffixes are stripped before searching.
    """
    cleaned = title
    for sep in ("|", " - ", "–", "—"):
        if sep in cleaned:
            cleaned = cleaned.split(sep)[0]
    for noise in ("official trailer", "final trailer", "teaser trailer",
                  "official teaser", "trailer", "teaser", "4k", "hd"):
        cleaned = cleaned.lower().replace(noise, "")
    cleaned = " ".join(cleaned.split()).strip(" -:|")
    if not cleaned:
        return None

    params = {"query": cleaned, "include_adult": "false"}
    if year:
        params["year"] = str(year)[:4]
    results = _get("/search/movie", **params).get("results") or []
    if not results and year:
        results = _get("/search/movie", query=cleaned,
                       include_adult="false").get("results") or []
    return results[0] if results else None


def get_cast(movie_id, limit=15):
    """Top-billed cast with portraits, in billing order."""
    credits = _get(f"/movie/{movie_id}/credits")
    cast = []
    for person in (credits.get("cast") or [])[:limit]:
        cast.append({
            "tmdb_id": person.get("id"),
            "name": person.get("name"),
            "character": person.get("character") or "",
            "profile_path": person.get("profile_path"),
            "portrait_url": portrait_url(person.get("profile_path")),
            "order": person.get("order", 999),
        })
    return cast


def enrich(title, year=None, limit=15):
    """Look a title up and return {movie, cast} -- the whole enrichment step.

    Returns None rather than raising when TMDB is unavailable or the title does
    not match, so ingestion can continue without metadata instead of failing.
    """
    try:
        movie = search_movie(title, year)
    except TMDBUnavailable:
        return None
    if not movie:
        return None

    try:
        cast = get_cast(movie["id"], limit=limit)
    except TMDBUnavailable:
        cast = []

    return {
        "movie": {
            "tmdb_id": movie.get("id"),
            "title": movie.get("title"),
            "year": (movie.get("release_date") or "")[:4],
            "synopsis": movie.get("overview") or "",
            "poster_path": movie.get("poster_path"),
        },
        "cast": cast,
    }


def download_portrait(url, dest_dir=None, filename=None):
    """Cache a portrait locally. Returns the path, or None on failure.

    Portraits are fetched at ingest and stored under a configured media
    directory. The database records only the servable URL, never this path --
    absolute paths used to leak to the browser through /api/config.
    """
    if not url:
        return None
    dest_dir = dest_dir or config.ACTOR_IMAGE_DIR
    os.makedirs(dest_dir, exist_ok=True)
    filename = filename or os.path.basename(urllib.parse.urlparse(url).path)
    dest = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SceneIt/0.1"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"  portrait download failed ({url}): {e}")
        return None
