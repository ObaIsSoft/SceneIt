import cv2
import numpy as np
import os
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Actor detection disabled.")

from numpy.linalg import norm
import requests
from PIL import Image
from io import BytesIO

class FaceDetector:
    def __init__(self):
        if not INSIGHTFACE_AVAILABLE:
            self.app = None
            self.known_faces = {}
            print("--- Face Model Disabled (Missing Dependency) ---")
            return

        print("--- Loading InsightFace (buffalo_l) ---")
        # On Mac M1/M2, CoreML or CPU is best. CUDA is not available.
        # We'll stick to CPU for compatibility/reliability.
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = {} # name -> embedding (numpy array)
        print("--- Face Model Loaded ---")

    def download_image(self, url):
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def register_actor(self, name, image_source):
        """
        Register an actor with a reference image (URL or local path or array).
        """
        img = None
        if isinstance(image_source, str):
            if image_source.startswith('http'):
                img = self.download_image(image_source)
            elif os.path.exists(image_source):
                img = cv2.imread(image_source)
        elif isinstance(image_source, np.ndarray):
            img = image_source
            
        if img is None:
            print(f"Could not load image for {name}")
            return False

        if self.app is None:
            print("Feature disabled: register_actor")
            return False

        faces = self.app.get(img)
        if not faces:
            print(f"No face found for {name}")
            return False
            
        # Take the largest face
        faces.sort(key=lambda x: x.bbox[2]*x.bbox[3], reverse=True)
        self.known_faces[name] = faces[0].embedding
        return True

    def scan_frame(self, image):
        """
        Returns a list of actor names found in the frame.
        """
        if self.app is None:
            return []

        faces = self.app.get(image)
        found_actors = []
        
        for face in faces:
            # Compare against all known actors
            for name, known_emb in self.known_faces.items():
                # Cosine Similarity
                # embeddings are already normalized by InsightFace? check.
                # Usually InsightFace embeddings are normalized.
                sim = np.dot(face.embedding, known_emb) / (norm(face.embedding) * norm(known_emb))
                if sim > 0.6: # Increased from 0.5 to 0.6 to reduce false positives (Production tuning)
                    found_actors.append(name)
                    break # Assume one identity per face
                    
        return list(set(found_actors))
