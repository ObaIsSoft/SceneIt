import os
import torch
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

# --- Configuration ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_frames")
METADATA_JSON = os.path.join(OUTPUT_DIR, "metadata.json")
# Vector index, saved as a compressed numpy archive. See vector_index.py.
INDEX_FILE = os.path.join(OUTPUT_DIR, "frames.npz")
FEEDBACK_FILE = "feedback_log.json"
TEMP_VIDEO_DIR = "temp_videos"

# --- AI Configuration ---
EMBEDDING_DIM = 512 # CLIP ViT-B/32 outputs 512 dimensions
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- YouTube API Configuration ---
# Get your API key from: https://console.cloud.google.com/apis/credentials
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_SEARCH_QUERIES = [
    "official movie trailer 2024",
    "official movie trailer 2023",
    "blockbuster movie trailer"
]

# --- Video Library ---
VIDEOS_TO_PROCESS = []
# Pre-defined videos (Commented out to speed up dev/demo)
VIDEOS_TO_PROCESS = [
    {
        "title": "BigBuckBunny",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "filename": "BigBuckBunny_CDN.mp4",
        "year": "2008",
        "synopsis": "A giant rabbit with a heart bigger than himself wakes up to start his day.",
        "actors": [
            {"name": "Big Buck Bunny", "character": "Himself", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Big_buck_bunny_poster_big.jpg/440px-Big_buck_bunny_poster_big.jpg"},
            {"name": "Frank", "character": "The Squirrel", "image": "https://peach.blender.org/wp-content/uploads/frank.png"}
        ],
        "streaming_on": [
            {"name": "YouTube", "url": "https://www.youtube.com/watch?v=YE7VzlLtp-4"},
            {"name": "Archive.org", "url": "https://archive.org/details/BigBuckBunny_328"}
        ]
    },
    {
        "title": "ElephantsDream",  
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "filename": "ElephantsDream_CDN.mp4",
        "year": "2006",
        "synopsis": "Emo and Proog explore a giant machine that is the world they live in.",
        "actors": [
            {"name": "Tygo Gernandt", "character": "Proog", "image": "https://upload.wikimedia.org/wikipedia/commons/9/90/Tygo-Gernandt.jpg"},
            {"name": "Cas Jansen", "character": "Emo", "image": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Cas_Jansen_2012.jpg"}
        ],
        "streaming_on": [
            {"name": "YouTube", "url": "https://www.youtube.com/watch?v=TLkA0RELQ1g"},
            {"name": "Archive.org", "url": "https://archive.org/details/ElephantsDream"}
        ]
    },
    {
        "title": "TearsOfSteel",
        "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
        "filename": "TearsOfSteel_CDN.mp4",
        "year": "2012",
        "synopsis": "A group of warriors and scientists gather at the Oude Kerk in Amsterdam to stage a desperate last stand against a robot apocalypse.",
        "actors": [
            {"name": "Derek de Lint", "character": "Thom", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Derek_de_Lint_De_Reunie_Premiere_2015.jpg/1200px-Derek_de_Lint_De_Reunie_Premiere_2015.jpg"},
            {"name": "Sergio Hasselbaink", "character": "Barney", "image": "https://m.media-amazon.com/images/M/MV5BMjA5OTE5MjA0Nl5BMl5BanBnXkFtZTgwNzY1Mjc2MTE@._V1_.jpg"},
            {"name": "Denise Rebergen", "character": "Celia", "image": "https://m.media-amazon.com/images/M/MV5BMTY5MjI0NDU5Nl5BMl5BanBnXkFtZTgwNjU4Mjc2MTE@._V1_.jpg"}
        ],
        "streaming_on": [
            {"name": "YouTube", "url": "https://www.youtube.com/watch?v=R6MlUcmOul8"},
            {"name": "Vimeo", "url": "https://vimeo.com/49719467"}
        ]
    },
    {
        "title": "Sintel",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
        "filename": "Sintel_CDN.mp4",
        "year": "2010",
        "synopsis": "A lonely young woman, Sintel, searches for a dragon she calls Scales.",
        "actors": [
             {"name": "Halina Reijn", "character": "Sintel", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Halina_Reijn.jpg/1200px-Halina_Reijn.jpg"},
             {"name": "Thom Hoffman", "character": "Shaman", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Thom_Hoffman_2012.jpg/1200px-Thom_Hoffman_2012.jpg"}
        ],
        "streaming_on": [
            {"name": "YouTube", "url": "https://www.youtube.com/watch?v=eRsGyueVLvQ"},
            {"name": "Archive.org", "url": "https://archive.org/details/Sintel"}
        ]
    },
    {
        "title": "ForBiggerBlazes",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "filename": "ForBiggerBlazes_CDN.mp4",
        "year": "2015",
        "synopsis": "A promotional video for Chromecast.",
        "actors": [],
        "streaming_on": [
            {"name": "YouTube", "url": "https://www.youtube.com/results?search_query=for+bigger+blazes"}
        ]
    },
    {
        "title": "MyVideo",
        "url": "",
        "filename": "IMG_6574.MOV",
        "year": "2024",
        "synopsis": "A personal video from the local library.",
        "actors": [],
        "streaming_on": [
            {"name": "Local Library", "url": "#"}
        ]
    }
]

# Load externally fetched trailers if available
import json
try:
    if os.path.exists("sceneit_trailers.json"):
        with open("sceneit_trailers.json", "r") as f:
            external_trailers = json.load(f)
            # Add default metadata for UI
            for t in external_trailers:
                if "year" not in t: t["year"] = "2024"
                if "synopsis" not in t: t["synopsis"] = f"Official trailer for {t['title']}."
                
                # --- MANUAL METADATA ENRICHMENT ---
                # Using local paths to avoid download errors
                actor_dir = os.path.abspath("actor_images")
                if "Dune" in t['title']:
                    t['actors'] = [
                        {"name": "Timothee Chalamet", "character": "Paul Atreides", "image": os.path.join(actor_dir, "timothee_chalamet.jpg")},
                        {"name": "Zendaya", "character": "Chani", "image": os.path.join(actor_dir, "zendaya.jpg")}
                    ]
                elif "Deadpool" in t['title']:
                    t['actors'] = [
                        {"name": "Ryan Reynolds", "character": "Deadpool", "image": os.path.join(actor_dir, "ryan_reynolds.jpg")},
                        {"name": "Hugh Jackman", "character": "Wolverine", "image": os.path.join(actor_dir, "hugh_jackman.jpg")},
                        {"name": "Emma Corrin", "character": "Cassandra Nova", "image": os.path.join(actor_dir, "emma_corrin.jpg")},
                        {"name": "Matthew Macfadyen", "character": "Paradox", "image": os.path.join(actor_dir, "matthew_macfadyen.jpg")}
                    ]
                elif "Godzilla" in t['title']:
                    t['actors'] = [
                         {"name": "Rebecca Hall", "character": "Dr. Ilene Andrews", "image": os.path.join(actor_dir, "rebecca_hall.jpg")}
                    ]
                else:
                    if "actors" not in t: t["actors"] = []
                # ----------------------------------

                if "streaming_on" not in t:
                    t["streaming_on"] = [{"name": "YouTube", "url": t['url']}]
            
            # Append to main list
            # Append to main list
            VIDEOS_TO_PROCESS.extend(external_trailers)
            print(f"Loaded {len(external_trailers)} external trailers from sceneit_trailers.json")
except Exception as e:
    print(f"Warning: Could not load sceneit_trailers.json: {e}")

FRAMES_TO_SKIP_SECONDS = 1.0 # Sample less frequently for vectors (slow extraction)
MIN_BRIGHTNESS = 20
MAX_BRIGHTNESS = 235
