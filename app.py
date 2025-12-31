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
import faiss

import config

# --- Global State Wrapper ---
class SearchEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.caption_model = None
        self.caption_processor = None
        self.faiss_index = None
        self.caption_cache = {}
        
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
        if os.path.exists(config.FAISS_INDEX_FILE):
             print(f"Loading Faiss index from {config.FAISS_INDEX_FILE}...")
             self.faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)
        else:
             print("--- Database not found. Initializing empty index. ---")
             self.faiss_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
             
        # 2. Load Metadata from SQLite
        # We load *everything* into memory to ensure 1:1 mapping with FAISS index integers
        # FAISS index 0 -> metadata_db[0]
        # We assume 'id' order in DB matches insertion order (which matches FAISS)
        try:
            from database import get_db_connection
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
            
            print(f"--- Database Ready: {self.faiss_index.ntotal} vectors, {len(self.metadata_db)} metadata rows. ---")
            
            # Sanity check
            if self.faiss_index.ntotal != len(self.metadata_db):
                print(f"Start-up Warning: Vector count ({self.faiss_index.ntotal}) != Metadata count ({len(self.metadata_db)})")
                
        except Exception as e:
            print(f"Error loading metadata from SQL: {e}")
            self.metadata_db = []

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
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize the vector (crucial for cosine similarity / L2 search)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
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
        # Disable SSL verify to avoid environment cert issues, enable timeout
        with requests.get(url, stream=True, headers=headers, timeout=30, verify=False) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        return dest_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def crop_black_bars(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, config.MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None: return img
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y+h, x:x+w]
        cv2.imwrite(image_path, cropped)
        return image_path
    except:
        return image_path

# --- Extraction Logic (V2) ---
# --- Extraction Logic (V2) ---
def run_extraction():
    # Reset everything
    # We will build the index at the END, after we know the total count
    all_vectors = []
    engine.metadata_db = [] # Clear memory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Force fresh DB schema
    if os.path.exists("sceneit.db"): os.remove("sceneit.db")
    from database import init_db, get_db_connection
    init_db()
    conn = get_db_connection()

    # Populate Videos Table first
    video_id_map = {}
    for vid in config.VIDEOS_TO_PROCESS:
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
    
    for video_info in config.VIDEOS_TO_PROCESS:
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
            cv2.imwrite(full_path, frame)
            
            # 3. Generate Vector Embedding
            vector = engine.get_embedding(frame)
            
            # 4. Face Detection
            found_actors = []
            if face_det:
                found_actors = face_det.scan_frame(frame)

            if vector is not None:
                # 5. Collect for later indexing
                all_vectors.append(vector)
                
                # Insert into DB (frames)
                cur = conn.execute('''
                    INSERT INTO frames (video_id, filename, timestamp, time_sec)
                    VALUES (?, ?, ?, ?)
                ''', (video_db_id, filename_jpg, timestamp_str, current_sec))
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

    # Convert list to huge numpy matrix
    print(f"\n--- Building Index for {total_vectors} vectors ---")
    vector_matrix = np.vstack(all_vectors)
    
    # Threshold for switching to IVF (Clustering)
    IVF_THRESHOLD = 50000 
    
    if total_vectors < IVF_THRESHOLD:
        print("Dataset is small. Using Exact Search (IndexFlatL2).")
        engine.faiss_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
        engine.faiss_index.add(vector_matrix)
    else:
        print(f"Dataset is large (> {IVF_THRESHOLD}). Using Clustering Search (IndexIVFFlat).")
        # Number of centroids (clusters)
        nlist = int(np.sqrt(total_vectors))
        quantizer = faiss.IndexFlatL2(config.EMBEDDING_DIM)
        engine.faiss_index = faiss.IndexIVFFlat(quantizer, config.EMBEDDING_DIM, nlist, faiss.METRIC_L2)
        
        print(f"Training index with {nlist} clusters...")
        engine.faiss_index.train(vector_matrix)
        print("Training complete. Adding vectors...")
        engine.faiss_index.add(vector_matrix)

    # Save Index & Metadata
    print(f"\nSaving Faiss index to {config.FAISS_INDEX_FILE}...")
    faiss.write_index(engine.faiss_index, config.FAISS_INDEX_FILE)
    
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
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    
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
        # 2. Crop
        crop_black_bars(temp_path)
        
        # 3. Generate Embedding
        query_img = cv2.imread(temp_path)
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
                    text_features = engine.model.get_text_features(**text_inputs)
                
                # Normalize text features
                text_features /= text_features.norm(dim=-1, keepdim=True)
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

        # 4. Search Faiss
        # We search MORE results if we have a filter, then filter them down
        search_k = 50 if actor_filter else 12 
        
        if isinstance(engine.faiss_index, faiss.IndexIVFFlat):
            engine.faiss_index.nprobe = 10 
            
        distances, indices = engine.faiss_index.search(query_vec, search_k)
        
        # Check for No Match
        # Heuristic: If the BEST match has a distance > 0.5 (arbitrary threshold for L2 normalized),
        # we consider it a failure. Note: for L2 on normalized vectors, dist = 2*(1-cos_sim).
        # cos_sim = 1 -> dist = 0. cos_sim = 0.8 -> dist = 0.4. 
        # So dist > 0.6 means similarity < 0.7.
        
        best_dist = distances[0][0]
        # Relax threshold slightly for multimodal (fusion noise)
        threshold = 0.75 if text_query else 0.65 
        
        if best_dist > threshold: 
             return jsonify({
                "match_type": "no_matches",
                "results": [],
                "query_hash": "vector_search"
            })
            
        results = []
        for i in range(search_k):
            idx = indices[0][i]
            dist = distances[0][i]
            
            if idx != -1 and idx < len(engine.metadata_db):
                meta = engine.metadata_db[idx]
                
                # Apply Actor Filter
                if actor_filter and meta['filename'] not in valid_filenames:
                    continue
                
                # RE-RANKING LOGIC - Boost previously selected results
                feedback_count = engine.feedback_db.get(meta['filename'], 0)
                adjusted_dist = float(dist) - (feedback_count * 0.08)  # Increased boost
                
                results.append({
                    "filename": meta['filename'],
                    "timestamp": meta['timestamp'],
                    "time_sec": meta['time_sec'],
                    "video_title": meta['video_title'],
                    "distance": adjusted_dist,
                    "is_boosted": feedback_count > 0,
                    "feedback_note": f"{feedback_count} votes" if feedback_count > 0 else "",
                    "detected_actors": meta.get('detected_actors', [])
                })
                
                if len(results) >= 12: break # Limit to 12 final results
        
        # Sort by adjusted distance
        results.sort(key=lambda x: x['distance'])
        
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
    if os.path.exists(config.FEEDBACK_FILE):
        try:
            with open(config.FEEDBACK_FILE, 'r') as f:
                log = json.load(f)
                # Filter for 'add' actions and get unique latest entries
                # Reverse to get newest first
                seen = set()
                for entry in reversed(log):
                    fname = entry.get('selected_filename')
                    action = entry.get('action', 'add')
                    if fname and action == 'add' and fname not in seen:
                        # Find metadata
                        meta = next((m for m in engine.metadata_db if m['filename'] == fname), None)
                        if meta:
                            history.append({
                                **meta, 
                                "distance": 0, # Dummy for frontend
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
                 random_clips.append({
                    **meta,
                    "distance": 0
                })

    return jsonify({
        "history": history,
        "random": random_clips
    })

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    
    if args.extract:
        run_extraction()
    else:
        if not os.path.exists(config.FAISS_INDEX_FILE):
            print("--- ERROR: Database not found. Run --extract first. ---")
        else:
            # Load models & DB on startup
            engine.load_ai_models()
            engine.load_database()
            engine.load_feedback_database()
            print("--- Server Started on Port 5002 ---")
            app.run(host='0.0.0.0', port=5002, debug=False)