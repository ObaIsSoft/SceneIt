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


# Minimum cosine similarity for a result to count as a match. Expressed as
# similarity so the number means something: 0.75 is "three-quarters aligned".
MATCH_THRESHOLD = 0.75
# Text fusion adds noise to the query vector, so accept slightly less.
MATCH_THRESHOLD_MULTIMODAL = 0.68


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
            # Refined Replacements matching prompt output
            if title:
                replacements = {}
                if title == "Sintel":
                    replacements = {
                        "a woman": "Sintel", "the woman": "Sintel", "a girl": "Sintel", 
                        "a dragon": "Scales", "the dragon": "Scales", "a small dragon": "Scales"
                    }
                elif title == "TearsOfSteel":
                    replacements = {
                        "a man": "Thom", "the man": "Thom", 
                        "a woman": "Celia", "the woman": "Celia",
                        "a robot": "a giant robot"
                    }
                elif title == "BigBuckBunny":
                    replacements = {
                        "a rabbit": "Big Buck Bunny", "the rabbit": "Big Buck Bunny", "a big rabbit": "Big Buck Bunny",
                         "a squirrel": "Frank", "three squirrels": "Frank and friends"
                    }
                elif title == "ElephantsDream":
                     replacements = {"a man": "Emo", "two men": "Emo and Proog"}

                for old, new in replacements.items():
                    # Case insensitive replace? No, BLIP output is lowercase usually
                    caption = caption.replace(old, new)
            
                    caption = caption.replace(old, new)
            
            self.caption_cache[image_path] = caption
            return caption

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
            
            # Iterate config to find match
            for vid in config.VIDEOS_TO_PROCESS:
                if vid['title'] == title_part:
                    return vid
        except:
            pass
        return None

    def _generate_clip_description(self, image_path):
        """Fast scene classification using CLIP (zero-shot)"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Scene categories (common in movies)
            scenes = [
                "outdoor scene", "indoor scene", "close-up shot", "wide shot",
                "action sequence", "dramatic scene", "conversation", 
                "dark scene", "bright scene", "cityscape", "nature scene"
            ]
            
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
            if confidence > 0.3:
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

def download_video_if_needed(url, dest_folder, dest_filename):
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, dest_filename)
    if os.path.exists(dest_path):
        print(f"Video {dest_filename} exists locally.")
        return dest_path
    print(f"Downloading {url}...")
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
    # Defaults to the whole configured library; --only / --limit narrow it so a
    # working index can be built in minutes instead of re-ingesting everything.
    if videos is None:
        videos = config.VIDEOS_TO_PROCESS
    if not videos:
        print("No videos selected for extraction.")
        return
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Force fresh DB schema.
    # Uses database.DB_PATH rather than a literal "sceneit.db": in production
    # DB_PATH points at a mounted volume (/data/sceneit.db on Fly), so the old
    # hardcoded path deleted nothing, init_db() ran CREATE TABLE IF NOT EXISTS
    # over the live database, and extraction appended to it -- duplicating
    # videos and desynchronising the index.
    from database import DB_PATH, get_db_connection, init_db
    if os.path.exists(DB_PATH):
        print(f"Removing existing database at {DB_PATH}")
        os.remove(DB_PATH)
    init_db()
    conn = get_db_connection()

    # Populate Videos Table first
    video_id_map = {}
    for vid in videos:
        metadata = {
            "synopsis": vid.get("synopsis"),
            "actors": vid.get("actors"),
            "streaming_on": vid.get("streaming_on")
        }
        cur = conn.execute('''
            INSERT INTO videos (title, filename, year, metadata_json)
            VALUES (?, ?, ?, ?)
        ''', (vid['title'], vid['filename'], vid['year'], json.dumps(metadata)))
        video_id_map[vid['title']] = cur.lastrowid
    conn.commit()

    # Ensure AI model is loaded
    if engine.model is None: engine.load_ai_models()
    
    for video_info in videos:
        title = video_info["title"]
        url = video_info["url"]
        filename = video_info["filename"]
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
                img_path = actor.get('image')
                db_image_url = img_path
                if img_path and "actor_images" in img_path:
                     basename = os.path.basename(img_path)
                     db_image_url = f"/api/actor_image/{basename}"

                conn.execute('''
                    INSERT OR REPLACE INTO actors (name, character, image) 
                    VALUES (?, ?, ?)
                ''', (actor['name'], actor.get('character'), db_image_url))
            conn.commit()
            
        except Exception as e:
            print(f"Face detection setup failed: {e}")
        
        video_path = download_video_if_needed(url, config.TEMP_VIDEO_DIR, filename)
        if not video_path: continue
        
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
    
    print(f"Saving metadata to {config.METADATA_JSON} (No longer authoritative, DB is used)...")
    # Optional: Keep writing JSON for debugging? Or delete?
    # Let's delete or comment out to avoid confusion.
    # with open(config.METADATA_JSON, 'w') as f:
    #    json.dump(engine.metadata_db, f, indent=4)
        
    print("--- Extraction Complete ---")

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
    return send_from_directory('actor_images', filename)

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
    return jsonify({
        "video_objects": config.VIDEOS_TO_PROCESS 
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
    parser = argparse.ArgumentParser(description="SceneIt server and ingest.")
    parser.add_argument('--extract', action='store_true',
                        help="rebuild the index from the configured library")
    parser.add_argument('--only', action='append', metavar='TITLE', default=None,
                        help="extract only this title (repeatable, substring match)")
    parser.add_argument('--limit', type=int, default=None, metavar='N',
                        help="extract at most N videos")
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 5002)))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--list', action='store_true',
                        help="list the configured library and exit")
    parser.add_argument('--video', metavar='URL_OR_PATH', default=None,
                        help="extract one arbitrary source, ignoring the configured library")
    parser.add_argument('--title', default=None, help="title for --video")
    parser.add_argument('--year', default=None, help="year for --video")
    args = parser.parse_args()

    if args.list:
        for i, v in enumerate(config.VIDEOS_TO_PROCESS):
            src = 'youtube' if 'youtube.com' in (v.get('url') or '') else ('local' if not v.get('url') else 'http')
            print(f"  [{i:2}] {v['title']:<46} {src:<8} {v.get('filename')}")
        raise SystemExit(0)

    if args.extract:
        selected = config.VIDEOS_TO_PROCESS

        # An explicit source bypasses the configured library entirely.
        if args.video:
            import re as _re
            from urllib.parse import urlparse as _urlparse
            is_url = args.video.startswith(('http://', 'https://'))
            basename = os.path.basename(_urlparse(args.video).path) or 'video.mp4'
            title = args.title or _re.sub(r'\.[^.]+$', '', basename) or 'Untitled'
            if not is_url and not os.path.exists(args.video):
                print(f"No such file: {args.video}")
                raise SystemExit(1)
            selected = [{
                "title": title,
                "url": args.video if is_url else "",
                "filename": basename if is_url else os.path.basename(args.video),
                "year": args.year or "",
                "synopsis": "",
                "actors": [],
                "streaming_on": [],
            }]
            # A local path is used in place, not re-downloaded.
            if not is_url:
                os.makedirs(config.TEMP_VIDEO_DIR, exist_ok=True)
                dest = os.path.join(config.TEMP_VIDEO_DIR, selected[0]["filename"])
                if os.path.abspath(dest) != os.path.abspath(args.video):
                    import shutil as _shutil
                    _shutil.copy2(args.video, dest)
        if args.only:
            wanted = [w.lower() for w in args.only]
            selected = [v for v in selected
                        if any(w in v['title'].lower() for w in wanted)]
            if not selected:
                print(f"No configured title matches {args.only}. Use --list to see them.")
                raise SystemExit(1)
        if args.limit is not None:
            selected = selected[:args.limit]
        print(f"--- Extracting {len(selected)} of {len(config.VIDEOS_TO_PROCESS)} videos ---")
        for v in selected:
            print(f"      {v['title']}")
        run_extraction(selected)
    else:
        if not os.path.exists(config.INDEX_FILE):
            print(f"--- ERROR: no index at {config.INDEX_FILE}. Build one first:")
            print("        python app.py --extract --only ForBiggerBlazes   (quick)")
            print("        python app.py --extract                          (full library)")
            raise SystemExit(1)

        # Load models & DB on startup
        engine.load_ai_models()
        engine.load_database()
        engine.load_feedback_database()
        print(f"--- Server listening on http://{args.host}:{args.port} ---")
        app.run(host=args.host, port=args.port, debug=False)