import os
import json
import datetime
import argparse
import cv2
import numpy as np
import requests
import tempfile
import random
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string

# --- NEW AI DEPENDENCIES ---
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from vector_index import VectorIndex

import config
import library


def new_index():
    """Build an empty index for cosine similarity over unit-length vectors.

    Two deliberate choices:

    * Cosine similarity rather than L2 distance. Embeddings are normalised,
      so the score is bounded in [-1, 1] and reads directly as a confidence.
      The old L2 distances needed magic cutoffs (0.65 / 0.75) and a frontend
      formula of `1 - d / 0.6` -- a guess layered on a guess.

    * Explicit ids, so a search returns the stored `frames.vector_id` instead
      of a row offset. See database.py for why that matters.

    See vector_index.py for why this is not FAISS.
    """
    return VectorIndex(config.EMBEDDING_DIM)


def _feature_tensor(features):
    """Normalise the return type of CLIPModel.get_image_features / get_text_features.

    transformers < 5 returned a plain tensor. transformers >= 5 returns a
    BaseModelOutputWithPooling, whose `pooler_output` is the projected
    embedding -- (1, 512) for ViT-B/32, matching config.projection_dim, while
    `last_hidden_state` is the unprojected (1, 50, 768) tower output.

    Without this, `features.norm(...)` raised AttributeError, get_embedding()
    swallowed it and returned None, and every search answered
    "Could not embed image" with a 500.
    """
    return getattr(features, "pooler_output", features)


# Match floors live in config so they can be tuned without a code change.
MATCH_THRESHOLD = config.MATCH_THRESHOLD
MATCH_THRESHOLD_MULTIMODAL = config.MATCH_THRESHOLD_MULTIMODAL


# --- Global State Wrapper ---
class SearchEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.caption_model = None
        self.caption_processor = None
        self.index = None
        self.caption_cache = {}

        # Ranking boosts loaded from the feedback log. Initialised here rather
        # than only in load_feedback_database(), because /api/search reads it
        # and used to raise AttributeError on any path that skipped the loader.
        self.feedback_db = {}

        # vector_id -> frame metadata. Replaces the old positional assumption
        # that FAISS row N corresponded to metadata_db[N]; see database.py.
        self.vector_map = {}
        
        # In-memory cache of DB (for fast FAISS lookup)
        # We load all frames from SQL -> List to match FAISS index ID
        self.metadata_db = [] 

    def load_ai_models(self):
        print(f"--- Loading AI Model ({config.MODEL_NAME}) on {config.DEVICE} ---")
        try:
            self.model = CLIPModel.from_pretrained(config.MODEL_NAME, use_safetensors=True).to(config.DEVICE)
            self.processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
            print("--- AI Model Loaded Successfully ---")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load AI model. {e}")
            exit(1)

    def load_caption_model(self):
        """Lazy load image captioning model with safetensors"""
        if self.caption_model is not None:
            return
        
        print("--- Loading Image Captioning Model ---")
        try:
            import warnings
            warnings.filterwarnings('ignore')
            
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Try loading with safetensors (bypasses torch version requirement)
            print("Attempting to load BLIP with safetensors...")
            self.caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                use_safetensors=True  # Use safetensors format
            ).to(config.DEVICE)
            
            print(f"✓ BLIP Model Loaded Successfully on {config.DEVICE}")
            return
            
        except Exception as e:
            print(f"⚠ BLIP failed: {e}")
            print("→ Falling back to CLIP-based scene classifier")
            
            # Fallback: Use CLIP for zero-shot scene classification
            self.caption_model = "clip_fallback"  # Marker
            self.caption_processor = None
            print("✓ Using CLIP for scene descriptions")

    # CACHE DISABLED FOR DEBUGGING
    # @lru_cache(maxsize=1000)
    def generate_caption(self, image_path):
        """Generate AI description for an image"""
        if image_path in self.caption_cache:
            print(f"[CACHE HIT] Serving caption for {os.path.basename(image_path)}")
            return self.caption_cache[image_path]

        if self.caption_model is None:
            self.load_caption_model()
        
        # If model failed to load completely
        if self.caption_model is None:
            return "Scene from video"
        
        try:
            # CLIP-based fallback (fast and reliable)
            if self.caption_model == "clip_fallback":
                clip_desc = self._generate_clip_description(image_path)
                # Attempt to extract context from filename
                context = self._get_context_from_filename(image_path)
                if context:
                    return f"{context} - {clip_desc}"
                return clip_desc
            
            # BLIP model (full captioning)
            image = Image.open(image_path).convert('RGB')
            
            # 1. Get Video Metadata
            video_info = self._get_video_info(image_path)
            title = video_info.get('title', '') if video_info else ''
            synopsis = video_info.get('synopsis', '') if video_info else ''
            
            # 2. Prompt Engineering
            # Use a more descriptive prompt trigger
            prompt = "a photography of "
            
            # BLIP model (full captioning)
            image = Image.open(image_path).convert('RGB')
            print(f"[DEBUG] PROMPT IS: {prompt}")

            # Pass text prompt to processor
            inputs = self.caption_processor(images=image, text=prompt, return_tensors="pt").to(config.DEVICE)
            
            # QUALITY OPTIMIZATION:
            # Removed min_length to prevent forced hallucination
            # Increased repetition_penalty slightly
            out = self.caption_model.generate(
                **inputs, 
                max_new_tokens=60, 
                num_beams=5, 
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
            
            # Decode
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            print(f"[DEBUG] RAW CAPTION OUTPUT: {caption}")
            
            # CLEANUP: Strip prompt
            if caption.lower().startswith(prompt.lower()):
                caption = caption[len(prompt):].strip()
            
            # Combine Context + AI Description
            # Context comes from config (manual metadata), Description comes from AI (visual)
            final_output = caption
            if synopsis:
                # Present it nicely: [Synopsis] ... [AI: ...]
                final_output = f"Context: {synopsis}\n\nAI Scene Detail: {caption}"
            elif title:
                final_output = f"Context: {title}\n\nAI Scene Detail: {caption}"
            
            return final_output

        # NOTE: ~30 lines of unreachable code sat here, after the return above.
        # It substituted character names into captions with per-title tables --
        # {"a woman": "Sintel"}, {"a man": "Thom"}, {"a rabbit": "Big Buck
        # Bunny"} -- and also held the only write to self.caption_cache, which
        # is why the cache never populated. It only ever covered the four
        # hardcoded demo films and has been removed rather than revived.

        except Exception as e:
            print(f"Caption error: {e}")
            return "Scene from video"

    def _get_context_from_filename(self, image_path):
        """Extracts title and timestamp from filename for context"""
        try:
            filename = os.path.basename(image_path)
            # Format: Title_Timestamp.jpg
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                title = parts[0]
                timestamp_str = parts[1].split('.')[0] # 150000
                seconds = int(timestamp_str) / 1000
                
                # Format time logic
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                time_fmt = f"{mins}:{secs:02d}"
                return f"{title} [{time_fmt}]"
        except:
            return None
        return None

    def _get_video_info(self, image_path):
        """Find matching config object for the image"""
        try:
            filename = os.path.basename(image_path) # TearsOfSteel_123.jpg
            title_part = filename.rsplit('_', 1)[0]
            
            # Look the title up in the library rather than scanning a
            # hardcoded Python list.
            return library.get_source(title_part)
        except:
            pass
        return None

    def _generate_clip_description(self, image_path):
        """Fast scene classification using CLIP (zero-shot)"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Labels come from config (SCENE_LABELS env var), because they
            # are content rather than logic.
            scenes = config.SCENE_LABELS
            
            # Use existing CLIP model
            inputs = self.processor(
                text=scenes,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(config.DEVICE)
            
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get top prediction
            top_idx = probs.argmax().item()
            confidence = probs[0][top_idx].item()
            
            # Return description
            scene_type = scenes[top_idx]
            if confidence > config.SCENE_LABEL_MIN_CONFIDENCE:
                return f"A {scene_type}"
            else:
                return "Scene from video"
                
        except Exception as e:
            print(f"CLIP description error: {e}")
            return "Scene from video"

    def load_database(self):
        # 1. Load FAISS Index
        if os.path.exists(config.INDEX_FILE):
             print(f"Loading vector index from {config.INDEX_FILE}...")
             self.index = VectorIndex.load(config.INDEX_FILE)
        else:
             print("--- Database not found. Initializing empty index. ---")
             self.index = new_index()

        # 2. Load Metadata from SQLite
        # Frames are keyed by their explicit vector_id, so a search result maps
        # back to a frame by lookup rather than by list position. The previous
        # version indexed metadata_db[] by FAISS row number, which silently
        # returned the wrong frame whenever the two drifted apart.
        try:
            from database import get_db_connection, migrate
            migrate()
            conn = get_db_connection()
            # Join with videos table to get 'video_title' comfortably
            rows = conn.execute('''
                SELECT f.*, v.title as video_title
                FROM frames f
                JOIN videos v ON f.video_id = v.id
                ORDER BY f.id ASC
            ''').fetchall()

            self.metadata_db = [dict(row) for row in rows]

            # 3. Enrich with Frame Detected Actors
            # Fetch all frame-actor links
            actor_rows = conn.execute('''
                SELECT fa.frame_id, a.name, a.character, a.image
                FROM frame_actors fa
                JOIN actors a ON fa.actor_id = a.id
            ''').fetchall()
            
            # Map frame_id -> list of actors
            frame_actors_map = {}
            for row in actor_rows:
                fid = row['frame_id']
                if fid not in frame_actors_map: frame_actors_map[fid] = []
                frame_actors_map[fid].append({
                    "name": row['name'],
                    "character": row['character'],
                    "image": row['image']
                })
            
            # Inject into metadata
            for meta in self.metadata_db:
                meta['detected_actors'] = frame_actors_map.get(meta['id'], [])

            conn.close()

            # Build the vector_id -> frame lookup that search results resolve
            # through. Frames with no embedding yet are simply absent from it.
            self.vector_map = {
                meta['vector_id']: meta
                for meta in self.metadata_db
                if meta.get('vector_id') is not None
            }

            print(
                f"--- Database Ready: {self.index.ntotal} vectors, "
                f"{len(self.metadata_db)} frames, {len(self.vector_map)} mapped. ---"
            )

            # Integrity check. This used to print a warning and carry on, which
            # meant a desynchronised index served wrong-but-plausible results
            # indefinitely. A mismatch now reports how to fix it, because the
            # only safe remedy is a rebuild.
            if self.index.ntotal != len(self.vector_map):
                print(
                    f"!!! INDEX INTEGRITY ERROR: {self.index.ntotal} vectors in the "
                    f"index but {len(self.vector_map)} frames claim one. Results would be "
                    f"unreliable. Rebuild with: python app.py --extract"
                )

        except Exception as e:
            print(f"Error loading metadata from SQL: {e}")
            self.metadata_db = []
            self.vector_map = {}

    def load_feedback_database(self):
        self.feedback_db = {}
        if not os.path.exists(config.FEEDBACK_FILE):
            return
            
        try:
            with open(config.FEEDBACK_FILE, 'r') as f:
                log = json.load(f)
                
            for entry in log:
                fname = entry.get('selected_filename')
                action = entry.get('action', 'add') # Default to add
                if fname:
                    if action == 'add':
                        self.feedback_db[fname] = self.feedback_db.get(fname, 0) + 1
                    elif action == 'remove':
                        self.feedback_db[fname] = max(0, self.feedback_db.get(fname, 0) - 1)
                        
            print(f"Loaded feedback counts for {len(self.feedback_db)} frames.")
        except Exception as e:
            print(f"Error loading feedback: {e}")

    def get_embedding(self, image_array):
        """
        Generates a 512-dim vector for a given CV2 image array.
        """
        try:
            # Convert CV2 (BGR) to PIL (RGB)
            img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            
            # Preprocess and run through model
            with torch.no_grad():
                inputs = self.processor(text=None, images=img, return_tensors="pt", padding=True).to(config.DEVICE)
                # Get image features
                image_features = _feature_tensor(self.model.get_image_features(**inputs))

            # Normalize, so the inner product against the index is the cosine.
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy float32 for Faiss
            return image_features.cpu().numpy().astype('float32')
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

# Initialize Global Engine
engine = SearchEngine()
app = Flask(__name__, static_folder='static')

# --- Helpers ---
def validate_image(file_stream):
    """
    Validates that the uploaded file is actually an image using magic bytes (via PIL).
    """
    try:
        img = Image.open(file_stream)
        img.verify() # Verify it's an image
        file_stream.seek(0) # Reset stream pointer
        return True
    except:
        return False

def download_with_ytdlp(url, dest_path, max_height=720):
    """Fetch a YouTube source with yt-dlp.

    Needed because plain requests on a watch URL saves the HTML page as .mp4.
    That is what the three configured trailers were doing: cv2 could not open
    the result, so they contributed zero frames and failed silently.

    Uses the Python API rather than the console script. shutil.which("yt-dlp")
    only finds it when the virtualenv's bin is on PATH, which it is not when
    invoking .venv/bin/python directly -- so the subprocess form reported
    "yt-dlp not found" despite yt-dlp being installed.
    """
    fmt = (f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/"
           f"best[ext=mp4][height<={max_height}]/best")

    try:
        import yt_dlp
    except ImportError:
        print("  yt-dlp is not installed. Add it with: pip install -e '.[ingest]'")
        return None

    options = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": dest_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  yt-dlp failed: {str(e).strip().splitlines()[-1][:200]}")
        return None

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  downloaded {size_mb:.1f} MB")
        return dest_path
    # yt-dlp may append a container extension when it cannot honour outtmpl.
    stem = os.path.splitext(dest_path)[0]
    for candidate in (f"{stem}.mp4", f"{stem}.mkv", f"{stem}.webm"):
        if os.path.exists(candidate):
            os.replace(candidate, dest_path)
            return dest_path
    print("  yt-dlp reported success but produced no file")
    return None


def download_video_if_needed(url, dest_folder, dest_filename, source_type=None):
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, dest_filename)
    if os.path.exists(dest_path):
        print(f"Video {dest_filename} exists locally.")
        return dest_path

    if source_type is None:
        source_type = library.classify_source(url)

    if source_type == "local":
        return dest_path if os.path.exists(dest_path) else None

    print(f"Downloading {url}...")
    if source_type == "youtube":
        return download_with_ytdlp(url, dest_path)

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        # verify=False was here, which disabled certificate validation on every
        # download and made the transfer trivially interceptable.
        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        return dest_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def normalize_frame(image):
    """Strip uniform dark borders (letterboxing) from a frame.

    MUST be applied identically at ingest and at query time. It previously was
    not: extraction stored frames untouched while /api/search cropped the
    uploaded image, so queries and index vectors lived in different visual
    spaces. On the Sintel trailer -- 1280x720 with baked-in 2.35:1 letterboxing
    -- cropping removed 175 of 720 rows, and a frame matched against its own
    stored vector at 0.9404 instead of 0.9989. That was enough for the
    neighbouring frame to outrank the correct one, so every search lost
    accuracy by a margin nothing was measuring.

    Takes and returns an array. The old version rewrote the file in place,
    which made the mutation invisible at the call site.
    """
    try:
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, config.MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        if w < 16 or h < 16:  # refuse to crop away the whole frame
            return image
        return image[y:y + h, x:x + w]
    except Exception as e:
        print(f"normalize_frame failed, using the frame as-is: {e}")
        return image

# --- Extraction Logic (V2) ---
# --- Extraction Logic (V2) ---
def run_extraction(videos=None):
    # Reset everything
    # We will build the index at the END, after we know the total count
    all_vectors = []
    engine.metadata_db = [] # Clear memory
    # Defaults to every source in the library; --only / --limit narrow it.
    if videos is None:
        videos = library.list_sources()
    if not videos:
        print("The library is empty. Add a source first:")
        print("    python app.py --add <url-or-path> --title 'Some Film'")
        return
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # The database is the library now, so it must NOT be deleted here. The old
    # code removed sceneit.db and repopulated `videos` from the hardcoded list
    # on every run; doing that today would destroy the catalogue. It also
    # removed a literal "sceneit.db" while DB_PATH may point at a mounted
    # volume, so in production it deleted nothing and appended duplicates
    # instead.
    from database import get_db_connection, init_db, migrate
    init_db()
    migrate()
    conn = get_db_connection()

    # This rebuilds the whole index, so every vector is recomputed. Frames are
    # therefore cleared wholesale to preserve the invariant that the index
    # contains exactly the frames the database knows about. Sources not in this
    # run return to 'pending' rather than being left claiming frames that no
    # longer exist.
    selected_ids = [v["id"] for v in videos]
    placeholders = ",".join("?" * len(selected_ids))
    dropped = conn.execute(
        f"SELECT COUNT(*) FROM videos WHERE status = ? AND id NOT IN ({placeholders})",
        [library.STATUS_INGESTED, *selected_ids],
    ).fetchone()[0]
    if dropped:
        print(f"Note: {dropped} previously ingested source(s) are not in this run "
              f"and will be reset to pending.")

    conn.execute("DELETE FROM frame_actors")
    conn.execute("DELETE FROM frames")
    conn.execute(
        f"UPDATE videos SET status = ?, ingested_at = NULL WHERE id NOT IN ({placeholders})",
        [library.STATUS_PENDING, *selected_ids],
    )
    conn.commit()

    video_id_map = {v["title"]: v["id"] for v in videos}

    # Ensure AI model is loaded
    if engine.model is None: engine.load_ai_models()
    
    for video_info in videos:
        title = video_info["title"]
        url = video_info.get("source_url") or ""
        filename = video_info.get("filename") or library.safe_filename(title, url)
        source_type = video_info.get("source_type") or library.classify_source(url)
        video_db_id = video_id_map[title]

        # 0. Face Recognition Setup
        face_det = None
        try:
            from face_id import FaceDetector
            face_det = FaceDetector()
            
            # Register actors for this video
            actors = video_info.get("actors", [])
            print(f"Registering {len(actors)} actors...")
            for actor in actors:
                # actor = {name, character, image}
                if 'image' in actor and actor['image']:
                     face_det.register_actor(actor['name'], actor['image'])
                    
            # Ensure actors exist in DB
            for actor in actors:
                # Convert abs path to URL for DB (if local)
                # Store a servable URL, never a filesystem path. Absolute
                # paths used to reach the browser via /api/config and rendered
                # as broken images while leaking the server layout.
                img_path = actor.get('image')
                db_image_url = img_path
                if img_path and not str(img_path).startswith(('http://', 'https://', '/api/')):
                     db_image_url = f"/api/actor_image/{os.path.basename(img_path)}"

                conn.execute('''
                    INSERT OR REPLACE INTO actors (name, character, image) 
                    VALUES (?, ?, ?)
                ''', (actor['name'], actor.get('character'), db_image_url))
            conn.commit()
            
        except Exception as e:
            print(f"Face detection setup failed: {e}")
        
        video_path = download_video_if_needed(url, config.TEMP_VIDEO_DIR, filename, source_type)
        if not video_path:
            library.mark_failed(video_db_id, f"could not fetch {url or filename}")
            continue
        
        print(f"\n--- Extracting Vectors for: {title} ---")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_skip = int(fps * config.FRAMES_TO_SKIP_SECONDS) or 1
        
        frame_count = 0
        processed_count = 0
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Filter dark/bright frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if not (config.MIN_BRIGHTNESS < brightness < config.MAX_BRIGHTNESS):
                frame_count += frame_skip
                continue
            
            # 2. Save Image Frame
            current_sec = frame_count / fps
            timestamp_str = str(datetime.timedelta(seconds=int(current_sec)))
            safe_ts = f"{int(current_sec * 1000)}"
            filename_jpg = f"{title}_{safe_ts}.jpg"
            full_path = os.path.join(config.OUTPUT_DIR, filename_jpg)
            # Same normalisation the query gets. Stored and embedded alike, so
            # the saved thumbnail is exactly what produced the vector.
            frame = normalize_frame(frame)
            cv2.imwrite(full_path, frame)
            
            # 3. Generate Vector Embedding
            vector = engine.get_embedding(frame)
            
            # 4. Face Detection
            found_actors = []
            if face_det:
                found_actors = face_det.scan_frame(frame)

            if vector is not None:
                # 5. Collect for later indexing, recording the id this vector
                # will be stored under. Written to the frame row in the same
                # transaction, so the mapping is persisted rather than inferred.
                vector_id = len(all_vectors)
                all_vectors.append(vector)

                # Insert into DB (frames)
                cur = conn.execute('''
                    INSERT INTO frames (video_id, filename, timestamp, time_sec, vector_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (video_db_id, filename_jpg, timestamp_str, current_sec, vector_id))
                frame_id = cur.lastrowid
                
                # Insert into DB (frame_actors)
                for actor_name in found_actors:
                    row = conn.execute("SELECT id FROM actors WHERE name = ?", (actor_name,)).fetchone()
                    if row:
                        conn.execute("INSERT INTO frame_actors (frame_id, actor_id) VALUES (?, ?)", (frame_id, row['id']))
                
                processed_count += 1
                
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} vectors...", end='\r')
            
            frame_count += frame_skip
            
        cap.release()
        conn.commit() # Commit per video
        if processed_count:
            library.mark_ingested(video_db_id, duration_sec=frame_count / fps if fps else None)
        else:
            library.mark_failed(video_db_id, "no frames passed the brightness filter")
        print(f"Finished {title}: {processed_count} vectors extracted.")

    # --- Dynamic Index Building ---
    total_vectors = len(all_vectors)
    if total_vectors == 0:
        print("No vectors extracted. Exiting.")
        return

    # Convert list to one matrix
    print(f"\n--- Building Index for {total_vectors} vectors ---")
    vector_matrix = np.vstack(all_vectors)

    # Exact cosine search with explicit ids. The ids are the same values written
    # to frames.vector_id above, in the same order, so index and database agree
    # by construction.
    #
    # The old IVF branch above 50k vectors is gone: it was untested at that
    # scale, needed an nprobe that nothing tuned, and exact search over a few
    # thousand 512-d vectors is a sub-millisecond matrix product. Approximate
    # search can come back when the corpus actually warrants it.
    engine.index = new_index()
    vector_ids = np.arange(total_vectors, dtype='int64')
    engine.index.add_with_ids(vector_matrix, vector_ids)

    # Verify before writing anything to disk.
    assert engine.index.ntotal == total_vectors, (
        f"index holds {engine.index.ntotal} vectors, expected {total_vectors}"
    )
    db_mapped = conn.execute(
        "SELECT COUNT(*) FROM frames WHERE vector_id IS NOT NULL"
    ).fetchone()[0]
    if db_mapped != total_vectors:
        print(
            f"!!! ABORT: {db_mapped} frames carry a vector_id but {total_vectors} vectors "
            f"were built. Not writing a desynchronised index."
        )
        conn.close()
        return
    print(f"Verified: {total_vectors} vectors <-> {db_mapped} frame rows.")

    # Save Index & Metadata
    print(f"\nSaving vector index to {config.INDEX_FILE}...")
    engine.index.save(config.INDEX_FILE)
    
    # metadata.json is gone. It was a second, stale copy of what the database
    # already holds -- the comment here read "No longer authoritative" while the
    # code still printed about writing it, and add_video.py still appended to
    # it. One source of truth.
    conn.close()
    print(f"--- Extraction Complete: {total_vectors} vectors ---")

# --- Flask Routes ---

@app.route('/')
def index():
    try:
        with open("index.html", "r") as f:
            return render_template_string(f.read())
    except: return "Error: index.html not found"

@app.route('/output_frames/<path:filename>')
def serve_image(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)

@app.route('/api/actor_image/<path:filename>')
def serve_actor_image(filename):
    return send_from_directory(config.ACTOR_IMAGE_DIR, filename)

@app.route('/api/video/<path:filename>')
def serve_video(filename):
    # Map friendly filename (from metadata) to actual file on disk
    video_path = os.path.join(config.TEMP_VIDEO_DIR, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    return send_from_directory(config.TEMP_VIDEO_DIR, filename)

@app.route('/api/caption', methods=['GET'])
def get_caption():
    """Generate AI caption for a frame"""
    from urllib.parse import unquote
    filename = request.args.get('filename')
    if filename: filename = unquote(filename)
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    
    frame_path = os.path.join(config.OUTPUT_DIR, filename)
    print(f"[DEBUG] Generating caption for: {frame_path}")
    
    if not os.path.exists(frame_path):
        print(f"[ERROR] Frame file not found: {frame_path}")
        return jsonify({"description": "Frame not found."})
    
    try:
        description = engine.generate_caption(frame_path)
        print(f"[DEBUG] Caption Generated: {description}")
        return jsonify({"description": description})
    except Exception as e:
        print(f"[ERROR] Caption generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"description": "AI description unavailable."})
        
@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Returns public configuration, specifically the list of available video titles.
    """
    sources = [library.to_api_dict(v) for v in library.list_sources()]
    return jsonify({
        # Kept under the original key so the frontend contract is unchanged.
        "video_objects": sources
    })

@app.route('/api/library', methods=['GET'])
def library_list():
    """The catalogue. Replaces reading a hardcoded Python list."""
    status = request.args.get('status')
    sources = library.list_sources(status=status)
    return jsonify({
        "sources": [library.to_api_dict(v) for v in sources],
        "counts": {
            "total": len(sources),
            "ingested": sum(1 for v in sources if v.get('status') == library.STATUS_INGESTED),
            "pending": sum(1 for v in sources if v.get('status') == library.STATUS_PENDING),
            "failed": sum(1 for v in sources if v.get('status') == library.STATUS_FAILED),
        },
    })


@app.route('/api/library', methods=['POST'])
def library_add():
    """Register a source. Adding is cheap: nothing downloads until ingest."""
    data = request.get_json(silent=True) or {}
    source = (data.get('source') or data.get('url') or '').strip()
    if not source:
        return jsonify({"error": "a 'source' url or path is required"}), 400

    try:
        row, created = library.add_source(
            source,
            title=data.get('title'),
            year=data.get('year'),
            synopsis=data.get('synopsis'),
            replace=bool(data.get('replace')),
        )
    except FileNotFoundError:
        return jsonify({"error": f"no such file: {source}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "source": library.to_api_dict(row),
        "created": created,
        "note": None if created else "a source with this title already exists; pass replace=true to overwrite",
    }), (201 if created else 200)


@app.route('/api/library/<int:source_id>', methods=['DELETE'])
def library_remove(source_id):
    removed = library.remove_source(source_id)
    if not removed:
        return jsonify({"error": "not found"}), 404
    # Its frames are gone by cascade, so their vectors are now orphaned.
    return jsonify({
        "removed": source_id,
        "reindex_required": True,
        "hint": "run: python app.py --extract",
    })


@app.route('/api/all_frames', methods=['GET'])
def get_all_frames():
    video_title = request.args.get('video_title')

    # Bare int() here raised ValueError -> 500 on any non-numeric input, so
    # ?page=abc was an unhandled server error. Clamp rather than reject: these
    # are pagination hints, not meaningful user input.
    def _positive_int(name, default, maximum):
        raw = request.args.get(name, default)
        try:
            return max(1, min(int(raw), maximum))
        except (TypeError, ValueError):
            return default

    page = _positive_int('page', 1, 1_000_000)
    limit = _positive_int('limit', 20, 100)
    
    # If no title provided, return randomly from all? Or just empty?
    # Let's support "all" if title is empty or "all"
    
    if not video_title or video_title == "all":
        frames = engine.metadata_db
    else:
        frames = [m for m in engine.metadata_db if m['video_title'] == video_title]
        
    # Pagination
    start = (page - 1) * limit
    end = start + limit
    paginated_frames = frames[start:end]
    
    return jsonify({
        "frames": paginated_frames,
        "total": len(frames),
        "has_more": end < len(frames)
    })

@app.route('/api/search', methods=['POST'])
def search():
    if 'query_image' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['query_image']
    
    # VALIDATION: Check if it's a real image
    if not validate_image(file):
        return jsonify({"error": "Invalid image file"}), 400

    # 1. Save Temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        file.save(temp.name)
        temp_path = temp.name
        
    try:
        # 2. Normalise exactly as ingest does, so query and index match.
        query_img = normalize_frame(cv2.imread(temp_path))

        # 3. Generate Embedding
        query_vec = engine.get_embedding(query_img)
        
        if query_vec is None:
            return jsonify({"error": "Could not embed image"}), 500
            
        # --- Multimodal Fusion ---
        text_query = request.form.get('text_query')
        if text_query and text_query.strip():
            print(f"[Search] Refining with text: '{text_query}'")
            try:
                # Get text features
                text_inputs = engine.processor(text=[text_query], return_tensors="pt", padding=True).to(config.DEVICE)
                with torch.no_grad():
                    text_features = _feature_tensor(engine.model.get_text_features(**text_inputs))

                # Normalize text features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_vec = text_features.cpu().numpy().astype('float32')
                
                # Weighted Combination (Image is still primary)
                # alpha = 0.7 (Image), beta = 0.3 (Text)
                query_vec = (query_vec * 0.7) + (text_vec * 0.3)
                
                # Re-normalize combined vector
                norm = np.linalg.norm(query_vec)
                if norm > 0: query_vec /= norm
                
            except Exception as e:
                print(f"Error processing text query: {e}")
                
        # --- Actor Filter ---
        actor_filter = request.form.get('actor_name')
        valid_indices = None
        if actor_filter:
            print(f"[Search] Filtering for actor: {actor_filter}")
            try:
                from database import get_db_connection
                conn = get_db_connection()
                # Get all frame filenames (or IDs) that have this actor
                rows = conn.execute('''
                    SELECT f.filename 
                    FROM frames f
                    JOIN frame_actors fa ON f.id = fa.frame_id
                    JOIN actors a ON fa.actor_id = a.id
                    WHERE a.name LIKE ?
                ''', (f"%{actor_filter}%",)).fetchall()
                valid_filenames = set(r['filename'] for r in rows)
                conn.close()
                print(f"Found {len(valid_filenames)} frames with {actor_filter}")
            except Exception as e:
                print(f"Actor filter error: {e}")
                valid_filenames = set()

        # 4. Search the index.
        # Over-fetch when filtering by actor, since most hits get discarded.
        search_k = 50 if actor_filter else 12
        search_k = min(search_k, max(engine.index.ntotal, 1))

        # Inner product over unit vectors: higher is better, and the score is
        # cosine similarity in [-1, 1].
        scores, ids = engine.index.search(query_vec, search_k)

        threshold = MATCH_THRESHOLD_MULTIMODAL if text_query else MATCH_THRESHOLD
        best_score = float(scores[0][0]) if len(scores[0]) else -1.0

        if best_score < threshold:
            return jsonify({
                "match_type": "no_matches",
                "results": [],
                "best_similarity": round(best_score, 4),
                "threshold": threshold,
                "query_hash": "vector_search"
            })

        results = []
        for score, vector_id in zip(scores[0], ids[0]):
            # faiss pads short result sets with -1.
            if vector_id == -1:
                continue

            # Resolve by explicit id. A miss means the index references a frame
            # the database does not have, which is a real integrity problem --
            # so skip it rather than silently serving a neighbouring frame.
            meta = engine.vector_map.get(int(vector_id))
            if meta is None:
                print(f"[Search] Orphaned vector_id {int(vector_id)} not in database; skipping")
                continue

            if actor_filter and meta['filename'] not in valid_filenames:
                continue

            # Re-ranking: nudge previously chosen frames up. Additive on
            # similarity, so a boost raises the score rather than lowering a
            # distance, and the result stays inside the comparable range.
            feedback_count = engine.feedback_db.get(meta['filename'], 0)
            boosted = min(1.0, float(score) + feedback_count * 0.02)

            results.append({
                "filename": meta['filename'],
                "timestamp": meta['timestamp'],
                "time_sec": meta['time_sec'],
                "video_title": meta['video_title'],
                "similarity": round(float(score), 4),
                "score": round(boosted, 4),
                "is_boosted": feedback_count > 0,
                "feedback_note": f"{feedback_count} votes" if feedback_count > 0 else "",
                "detected_actors": meta.get('detected_actors', [])
            })

            if len(results) >= 12:
                break

        # Best match first.
        results.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({
            "match_type": "image_grid",
            "results": results,
            "query_hash": "vector_search" # Placeholder
        })
        
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    try:
        log = []
        if os.path.exists(config.FEEDBACK_FILE):
            with open(config.FEEDBACK_FILE, 'r') as f: log = json.load(f)
            
        action = data.get('action', 'add') # add or remove
            
        log.append({
            "selected_filename": data.get('selected_filename'),
            "query_hash": "vector_search",
            "action": action,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        with open(config.FEEDBACK_FILE, 'w') as f: json.dump(log, f)
        
        # Reload in-memory DB
        engine.load_feedback_database()
        return jsonify({"status": "success"})
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed"}), 500

@app.route('/api/explore', methods=['GET'])
def explore():
    # 1. History (Recent Selections from Feedback)
    history = []
    # Declared out here, not inside the branch below. It is read unconditionally
    # by the random-picks section, so on a fresh install with no feedback log
    # this endpoint raised NameError and returned a 500.
    seen = set()
    if os.path.exists(config.FEEDBACK_FILE):
        try:
            with open(config.FEEDBACK_FILE, 'r') as f:
                log = json.load(f)
                # Filter for 'add' actions and get unique latest entries
                # Reverse to get newest first
                for entry in reversed(log):
                    fname = entry.get('selected_filename')
                    action = entry.get('action', 'add')
                    if fname and action == 'add' and fname not in seen:
                        # Find metadata
                        meta = next((m for m in engine.metadata_db if m['filename'] == fname), None)
                        if meta:
                            history.append({
                                **meta,
                                "feedback_note": "Recently Found"
                            })
                            seen.add(fname)
                    if len(history) >= 10: break
        except Exception as e:
            print(f"Explore history error: {e}")

    # 2. Random Clips (Discovery)
    random_clips = []
    if engine.metadata_db:
        # Get random sample
        sample_size = min(20, len(engine.metadata_db))
        random_indices = random.sample(range(len(engine.metadata_db)), sample_size)
        for idx in random_indices:
            meta = engine.metadata_db[idx]
            # Don't duplicate history
            if meta['filename'] not in seen:
                 random_clips.append(dict(meta))

    return jsonify({
        "history": history,
        "random": random_clips
    })

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SceneIt: manage the library, build the index, serve the app.",
        epilog="The library lives in the database. Add sources, then extract.",
    )
    # Library management
    parser.add_argument('--add', metavar='URL_OR_PATH', action='append', default=None,
                        help="add a source to the library (repeatable)")
    parser.add_argument('--title', default=None, help="title for --add")
    parser.add_argument('--year', default=None, help="year for --add")
    parser.add_argument('--remove', metavar='ID_OR_TITLE', default=None,
                        help="remove a source from the library")
    parser.add_argument('--list', action='store_true', help="list the library and exit")

    # Indexing
    parser.add_argument('--extract', action='store_true',
                        help="rebuild the index from the library")
    parser.add_argument('--only', action='append', metavar='TITLE', default=None,
                        help="restrict extraction to matching titles (repeatable, substring)")
    parser.add_argument('--limit', type=int, default=None, metavar='N',
                        help="extract at most N sources")
    parser.add_argument('--pending', action='store_true',
                        help="extract only sources not yet ingested")

    # Serving
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 5002)))
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    from database import init_db, migrate
    init_db()
    migrate()

    def print_library():
        sources = library.list_sources()
        if not sources:
            print("  (library is empty)")
            return
        print(f"  {'ID':>3}  {'TITLE':<44} {'TYPE':<8} {'STATUS':<9} {'FRAMES':>6}")
        for v in sources:
            print(f"  {v['id']:>3}  {(v['title'] or '')[:44]:<44} "
                  f"{(v['source_type'] or '?'):<8} {(v['status'] or '?'):<9} "
                  f"{v.get('frame_count', 0):>6}")
            if v.get('status') == library.STATUS_FAILED and v.get('error'):
                print(f"       error: {v['error'][:90]}")

    # --- Library management -------------------------------------------------
    did_manage = False

    if args.add:
        did_manage = True
        if args.title and len(args.add) > 1:
            print("--title applies to a single --add; omit it to derive titles automatically.")
            raise SystemExit(2)
        for source in args.add:
            try:
                row, created = library.add_source(
                    source, title=args.title, year=args.year, replace=False)
            except FileNotFoundError:
                print(f"  no such file: {source}")
                raise SystemExit(1)
            except ValueError as e:
                print(f"  {e}")
                raise SystemExit(1)
            verb = "added" if created else "already present"
            print(f"  {verb}: [{row['id']}] {row['title']}  ({row['source_type']})")

    if args.remove:
        did_manage = True
        if library.remove_source(args.remove):
            print(f"  removed: {args.remove}  (run --extract to rebuild the index)")
        else:
            print(f"  not found: {args.remove}")
            raise SystemExit(1)

    if args.list:
        print_library()
        raise SystemExit(0)

    # --- Indexing -----------------------------------------------------------
    if args.extract:
        everything = library.list_sources()
        if not everything:
            print("The library is empty. Add a source first:")
            print("    python app.py --add <url-or-path> --title 'Some Film'")
            raise SystemExit(1)

        selected = everything
        if args.pending:
            selected = [v for v in selected if v.get('status') != library.STATUS_INGESTED]
        if args.only:
            wanted = [w.lower() for w in args.only]
            selected = [v for v in selected
                        if any(w in (v['title'] or '').lower() for w in wanted)]
            if not selected:
                print(f"No title in the library matches {args.only}.")
                print_library()
                raise SystemExit(1)
        if args.limit is not None:
            selected = selected[:args.limit]
        if not selected:
            print("Nothing to extract.")
            raise SystemExit(0)

        print(f"--- Extracting {len(selected)} of {len(everything)} source(s) ---")
        for v in selected:
            print(f"      [{v['id']}] {v['title']}")
        run_extraction(selected)
        raise SystemExit(0)

    if did_manage:
        print()
        print_library()
        raise SystemExit(0)

    # --- Serving ------------------------------------------------------------
    if not os.path.exists(config.INDEX_FILE):
        print(f"--- ERROR: no index at {config.INDEX_FILE}.")
        print("    1. add a source:  python app.py --add <url> --title 'Some Film'")
        print("    2. build it:      python app.py --extract")
        raise SystemExit(1)

    engine.load_ai_models()
    engine.load_database()
    engine.load_feedback_database()
    print(f"--- Server listening on http://{args.host}:{args.port} ---")
    app.run(host=args.host, port=args.port, debug=False)
