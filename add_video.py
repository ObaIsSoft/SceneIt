import os
import argparse
import cv2
import json
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import config

def add_local_video(video_path, title):
    print(f"--- Processing Local Video: {title} ---")
    
    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return

    # 1. Load AI API
    print("Loading AI Model...")
    model = CLIPModel.from_pretrained(config.MODEL_NAME, use_safetensors=True).to(config.DEVICE)
    processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
    
    # 2. Load Existing Database
    metadata_db = []
    if os.path.exists(config.METADATA_JSON):
        with open(config.METADATA_JSON, 'r') as f:
            metadata_db = json.load(f)
            
    index = None
    if os.path.exists(config.FAISS_INDEX_FILE):
        index = faiss.read_index(config.FAISS_INDEX_FILE)
    else:
        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)

    # 3. Process Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {duration:.1f}s, {fps:.1f} FPS")
    
    frame_interval = int(fps * config.FRAMES_TO_SKIP_SECONDS)
    current_frame = 0
    count = 0
    
    new_vectors = []
    new_metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if current_frame % frame_interval == 0:
            # Check brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if config.MIN_BRIGHTNESS < brightness < config.MAX_BRIGHTNESS:
                # Save Frame
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                filename = f"{title}_{timestamp_ms}.jpg".replace(" ", "_")
                save_path = os.path.join(config.OUTPUT_DIR, filename)
                cv2.imwrite(save_path, frame)
                
                # Embedding
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
                with torch.no_grad():
                    embed = model.get_image_features(**inputs).cpu().numpy()
                
                # Normalize
                faiss.normalize_L2(embed)
                new_vectors.append(embed[0])
                
                # Metadata
                new_metadata.append({
                    "filename": filename,
                    "video_title": title,
                    "timestamp": timestamp_ms,
                    "time_sec": timestamp_ms / 1000.0
                })
                count += 1
                if count % 10 == 0: print(f"Processed {count} frames...")
        
        current_frame += 1
        
    cap.release()
    
    # 4. Save Updates
    if new_vectors:
        print(f"Adding {len(new_vectors)} new vectors to index...")
        vectors_np = np.array(new_vectors).astype('float32')
        index.add(vectors_np)
        faiss.write_index(index, config.FAISS_INDEX_FILE)
        
        metadata_db.extend(new_metadata)
        with open(config.METADATA_JSON, 'w') as f:
            json.dump(metadata_db, f)
            
        print("--- Success! Video added to database. ---")
        print("Please restart app.py to see changes.")
    else:
        print("No valid frames extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to video file")
    parser.add_argument('--title', help="Title of the movie", default="Unknown Video")
    args = parser.parse_args()
    
    add_local_video(args.path, args.title)
