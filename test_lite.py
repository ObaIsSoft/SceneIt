"""
Lightweight tests for visual_search that avoid heavy model loading.
Run these first to verify the core application logic.
"""
import pytest
import json
from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- API Endpoint Tests (Lightweight) ---

class TestCoreAPI:
    """Test core API endpoints without heavy dependencies."""
    
    def test_index_route(self, client):
        """Test the index route returns HTML."""
        response = client.get('/')
        assert response.status_code == 200
    
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
        print(f"✓ Config loaded: {len(data['video_objects'])} videos")
    
    def test_search_endpoint_no_file(self, client):
        """Test search endpoint without file returns error."""
        response = client.post('/api/search')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        print(f"✓ Correctly rejects empty search")
    
    def test_caption_endpoint_no_filename(self, client):
        """Test caption endpoint without filename returns error."""
        response = client.get('/api/caption')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        print(f"✓ Correctly rejects caption request without filename")
    
    def test_caption_endpoint_invalid_file(self, client):
        """Test caption endpoint with non-existent file."""
        response = client.get('/api/caption?filename=nonexistent.jpg')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'description' in data
        assert data['description'] == 'Frame not found.'
        print(f"✓ Handles missing files gracefully")
    
    def test_all_frames_pagination(self, client):
        """Test /api/all_frames pagination structure."""
        response = client.get('/api/all_frames?video_title=test&page=1&limit=10')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'frames' in data
        assert 'total' in data
        assert 'has_more' in data
        assert isinstance(data['frames'], list)
        print(f"✓ Pagination endpoint structure correct")
    
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
        print(f"✓ Feedback endpoint accepts data")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
