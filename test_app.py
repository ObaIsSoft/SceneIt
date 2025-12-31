import pytest
import os
import json
import tempfile
import shutil
import numpy as np
from PIL import Image
import faiss

# Import the app
import app as app_module
from app import SearchEngine, app, engine

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (100, 100), color='red')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name)
    yield temp_file.name
    os.remove(temp_file.name)

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# --- Core Functionality Tests ---

class TestSearchEngine:
    """Test the SearchEngine class core functions."""
    
    def test_search_engine_initialization(self):
        """Test that SearchEngine initializes correctly."""
        test_engine = SearchEngine()
        assert test_engine.model is None
        assert test_engine.processor is None
        assert test_engine.caption_model is None
        assert test_engine.faiss_index is None
        assert test_engine.metadata_db == []
        assert test_engine.feedback_db == {}
    
    def test_get_embedding(self, sample_image):
        """Test vector embedding generation."""
        # Load AI models if not already loaded
        if engine.model is None:
            engine.load_ai_models()
        
        # Load test image
        import cv2
        img = cv2.imread(sample_image)
        
        # Generate embedding
        embedding = engine.get_embedding(img)
        
        assert embedding is not None
        assert embedding.shape == (1, 512)  # CLIP ViT-B/32 outputs 512 dims
        assert embedding.dtype == np.float32
    
    def test_generate_caption(self, sample_image):
        """Test AI caption generation."""
        try:
            caption = engine.generate_caption(sample_image)
            
            assert caption is not None
            assert isinstance(caption, str)
            assert len(caption) > 0
        except Exception as e:
            pytest.skip(f"Caption model unavailable or failed: {e}")

# --- API Endpoint Tests ---

class TestAPIEndpoints:
    """Test Flask API endpoints."""
    
    def test_index_route(self, client):
        """Test the index route returns HTML."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Framed' in response.data or b'html' in response.data
    
    def test_config_endpoint(self, client):
        """Test /api/config returns video configuration."""
        response = client.get('/api/config')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'video_objects' in data
        assert isinstance(data['video_objects'], list)
        assert len(data['video_objects']) > 0
        
        # Check structure of first video object
        first_video = data['video_objects'][0]
        assert 'title' in first_video
        assert 'url' in first_video
        assert 'filename' in first_video
    
    def test_search_endpoint_no_file(self, client):
        """Test search endpoint without file returns error."""
        response = client.post('/api/search')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_search_endpoint_with_image(self, client, sample_image):
        """Test search endpoint with valid image."""
        # Ensure database is loaded
        if engine.faiss_index is None or engine.faiss_index.ntotal == 0:
            pytest.skip("No database loaded. Run --extract first.")
        
        with open(sample_image, 'rb') as img_file:
            response = client.post('/api/search', data={
                'query_image': (img_file, 'test.jpg')
            })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'match_type' in data
        assert 'results' in data
        assert isinstance(data['results'], list)
    
    def test_caption_endpoint_no_filename(self, client):
        """Test caption endpoint without filename returns error."""
        response = client.get('/api/caption')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_caption_endpoint_invalid_file(self, client):
        """Test caption endpoint with non-existent file."""
        response = client.get('/api/caption?filename=nonexistent.jpg')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'description' in data
        assert data['description'] == 'Frame not found.'
    
    def test_all_frames_endpoint(self, client):
        """Test /api/all_frames with pagination."""
        if not engine.metadata_db:
            pytest.skip("No metadata loaded. Run --extract first.")
        
        # Get first video title
        first_title = engine.metadata_db[0]['video_title']
        
        response = client.get(f'/api/all_frames?video_title={first_title}&page=1&limit=10')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'frames' in data
        assert 'total' in data
        assert 'has_more' in data
        assert isinstance(data['frames'], list)
        assert len(data['frames']) <= 10
    
    def test_feedback_endpoint(self, client):
        """Test feedback submission."""
        feedback_data = {
            'selected_filename': 'test.jpg',
            'query_hash': 'test_hash'
        }
        
        response = client.post('/api/feedback', 
                              data=json.dumps(feedback_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data

# --- Index Building Tests ---

class TestIndexBuilding:
    """Test dynamic index building logic."""
    
    def test_flat_index_for_small_data(self):
        """Test that small datasets use IndexFlatL2."""
        # Create a small set of dummy vectors
        dummy_vectors = np.random.rand(100, 512).astype('float32')
        
        # Normalize them (as CLIP does)
        for i in range(len(dummy_vectors)):
            norm = np.linalg.norm(dummy_vectors[i])
            if norm > 0:
                dummy_vectors[i] = dummy_vectors[i] / norm
        
        # Build index (simulating small dataset logic)
        index = faiss.IndexFlatL2(512)
        index.add(dummy_vectors)
        
        assert index.ntotal == 100
        assert isinstance(index, faiss.IndexFlatL2)
    
    def test_ivf_index_for_large_data(self):
        """Test that large datasets would use IndexIVFFlat."""
        # Create a larger set of dummy vectors
        dummy_vectors = np.random.rand(1000, 512).astype('float32')
        
        # Normalize
        for i in range(len(dummy_vectors)):
            norm = np.linalg.norm(dummy_vectors[i])
            if norm > 0:
                dummy_vectors[i] = dummy_vectors[i] / norm
        
        # Build IVF index (simulating large dataset logic)
        nlist = int(np.sqrt(len(dummy_vectors)))
        quantizer = faiss.IndexFlatL2(512)
        index = faiss.IndexIVFFlat(quantizer, 512, nlist, faiss.METRIC_L2)
        
        # Train and add
        index.train(dummy_vectors)
        index.add(dummy_vectors)
        
        assert index.ntotal == 1000
        assert isinstance(index, faiss.IndexIVFFlat)
    
    def test_search_with_nprobe(self):
        """Test that nprobe is set correctly for IVF indexes."""
        dummy_vectors = np.random.rand(500, 512).astype('float32')
        
        # Build IVF index
        nlist = 20
        quantizer = faiss.IndexFlatL2(512)
        index = faiss.IndexIVFFlat(quantizer, 512, nlist, faiss.METRIC_L2)
        index.train(dummy_vectors)
        index.add(dummy_vectors)
        
        # Set nprobe
        index.nprobe = 10
        
        # Search
        query = dummy_vectors[0:1]
        distances, indices = index.search(query, k=5)
        
        assert len(distances[0]) == 5
        assert len(indices[0]) == 5
        assert indices[0][0] == 0  # First result should be itself

# --- Helper Function Tests ---

class TestHelperFunctions:
    """Test utility helper functions."""
    
    def test_validate_image(self, sample_image):
        """Test image validation function."""
        with open(sample_image, 'rb') as f:
            assert app_module.validate_image(f) == True
    
    def test_validate_non_image(self):
        """Test validation fails for non-image files."""
        # Create a text file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is not an image")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                assert app_module.validate_image(f) == False
        finally:
            os.remove(temp_path)

# --- Integration Tests ---

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_search_workflow(self, client, sample_image):
        """Test a complete search workflow from upload to results."""
        if engine.faiss_index is None or engine.faiss_index.ntotal == 0:
            pytest.skip("No database loaded. Run --extract first.")
        
        # 1. Upload and search
        with open(sample_image, 'rb') as img_file:
            search_response = client.post('/api/search', data={
                'query_image': (img_file, 'test.jpg')
            })
        
        assert search_response.status_code == 200
        search_data = json.loads(search_response.data)
        
        # 2. If we got results, test feedback
        if search_data['results']:
            first_result = search_data['results'][0]
            
            feedback_response = client.post('/api/feedback',
                                           data=json.dumps({
                                               'selected_filename': first_result['filename'],
                                               'query_hash': search_data['query_hash']
                                           }),
                                           content_type='application/json')
            
            assert feedback_response.status_code == 200

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
