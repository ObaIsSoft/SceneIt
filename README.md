# SceneIt

Find the exact movie scene a screenshot came from.

Upload a still — or a short clip — and SceneIt tells you which title it belongs to
and the timestamp it appears at. Frames are embedded into a vector index at
ingest time; a query image is embedded the same way and matched against it.

> **Status: mid-refactor.** Search works and the library is now data you manage
> at runtime. The remaining work is weight: the runtime still carries roughly
> 2 GB of model files it does not need. See [Roadmap](#roadmap).

## How it works

```
video ──▶ frame sampling ──▶ image encoder ──▶ vector index
                                                    │
screenshot ──▶ image encoder ──────────────────▶ nearest neighbours ──▶ title + timestamp
```

| Piece | Current implementation |
| --- | --- |
| Library | rows in SQLite, managed by CLI or `POST /api/library` |
| Frame sampling | one frame per second, filtered on mean brightness |
| Image encoder | CLIP ViT-B/32 (512-d), via `transformers` |
| Vector index | FAISS — exact `IndexFlatL2`, switching to `IndexIVFFlat` past 50k vectors |
| Metadata | SQLite (`videos`, `frames`, `actors`, `frame_actors`) |
| Actor recognition | InsightFace `buffalo_l`, cosine match against reference portraits |
| Text refinement | CLIP text tower, blended into the image vector |
| API + UI | Flask, server-rendered single page, vanilla JS |

## Requirements

- **Python 3.11 or 3.12.** Not 3.14 — the pinned numpy and OpenCV lines have no
  wheels for it yet.
- `ffmpeg` on `PATH`, for video decoding.
- Roughly 3 GB of disk for model weights on first run.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[torch-baseline,dev]'
```

Optional extras:

```bash
pip install -e '.[faces]'    # actor recognition (needs a C compiler)
pip install -e '.[ingest]'   # YouTube discovery and download
```

Then add something to search and build an index:

```bash
cp .env.example .env         # fill in your own keys

# 1. Add sources. Nothing downloads yet, so this is cheap and offline.
python app.py --add "https://www.youtube.com/watch?v=..." --title "Some Film" --year 2024
python app.py --add ./my-footage.mp4 --title "My Footage"

# 2. Build the index: fetches each source, samples frames, embeds them.
python app.py --extract

# 3. Serve.
python app.py
```

The library starts **empty**. There is no bundled demo content and no module
names a film — what gets indexed is entirely up to you.

### Managing the library

```bash
python app.py --list                     # catalogue, with status and frame counts
python app.py --extract --pending        # only sources not yet ingested
python app.py --extract --only dune      # substring match on title
python app.py --extract --limit 3
python app.py --remove 4                 # by id or exact title
```

YouTube sources need `yt-dlp` (the `ingest` extra); a plain HTTP fetch of a
watch URL saves the HTML page, not the video. Each source carries a status of
`pending`, `ingested` or `failed`, and a failed one records why.

To search YouTube for sources instead of pasting URLs:

```bash
python discover.py "official movie trailer 2026" --limit 5 --dry-run
```

## Configuration

All configuration is environment-driven. See [`.env.example`](.env.example) for
the full list.

| Variable | Purpose |
| --- | --- |
| `YOUTUBE_API_KEY` | Trailer discovery. Only needed for ingestion. |
| `TMDB_API_KEY` | Title metadata — synopsis, cast, year. Optional. |
| `DB_PATH` | SQLite location. Defaults to `sceneit.db`. |
| `OUTPUT_DIR` | Extracted frames and index. Defaults to `output_frames/`. |

Never put a real value in `.env.example` — it is committed. Secrets belong in
`.env`, which is ignored.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/search` | Match an uploaded image. Accepts optional `text_query` and `actor_name`. |
| `GET /api/all_frames` | Paginated frame listing, filterable by title. |
| `GET /api/caption` | Scene description for one frame. |
| `GET /api/explore` | Recently matched frames plus a random sample. |
| `POST /api/feedback` | Record a chosen result; boosts it in later rankings. |
| `GET /api/config` | Library contents, for the frontend. |
| `GET /api/library` | The catalogue, with per-status counts. |
| `POST /api/library` | Register a source (`{"source": "...", "title": "..."}`). |
| `DELETE /api/library/<id>` | Remove a source and its frames. |

## Tests

```bash
pytest              # everything
pytest -m unit      # fast, no model required
pytest -m api       # endpoint behaviour
```

Tests that need a built index are skipped when none is present, so a clean
checkout still gets a green run.

## Deployment

Containerised; see [DEPLOYMENT.md](DEPLOYMENT.md). [`fly.toml`](fly.toml) is
configured for Fly.io with a persistent volume at `/data`, which keeps the
database and index from being wiped on restart.

Note that `temp_videos/` is excluded from the image, so clip playback degrades
to a still frame in a deployed container.

## Roadmap

The refactor in progress, in order:

1. ~~**Index integrity**~~ — done. The frame-to-vector mapping is explicit, so a
   partial re-ingest can no longer return the wrong frame silently.
2. ~~**De-hardcoding**~~ — done. The library is rows in SQLite fed by a CLI and
   an ingest API. No module names a film.
3. **Lighter runtime** — drop `torch` and `transformers` for ONNX Runtime, and
   swap CLIP for a copy-detection descriptor, which is what screenshot matching
   actually calls for. Target is roughly 260 MB of runtime, down from about 2.5 GB.
4. **Shot-aware ingest** — sample on detected shot boundaries rather than on a
   fixed one-second interval, and align clip queries temporally so a match
   reports a span rather than a single frame.
5. **Faces by track** — aggregate face embeddings across a track and cluster
   them, so cast is discovered from the footage instead of from hand-placed
   reference portraits.

Measured on three real trailers (335 frames): the correct film is returned at
rank 1 for 100% of indexed frames, and the exact frame for 98.5%. The five
inexact hits are genuinely duplicate opening frames that all score identically
— which is what item 4 addresses.

## Licence

Not yet chosen. Note that model weights carry their own terms, separate from
this code.
