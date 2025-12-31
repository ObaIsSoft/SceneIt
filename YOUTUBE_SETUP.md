# SceneIt - YouTube API Setup Guide

## Step 1: Get YouTube API Key (Free)

1. **Go to Google Cloud Console**:
   - Visit: https://console.cloud.google.com/

2. **Create a New Project**:
   - Click "Select a Project" → "New Project"
   - Name it "SceneIt" or similar
   - Click "Create"

3. **Enable YouTube Data API v3**:
   - In the search bar, type "YouTube Data API v3"
   - Click on it → Click "Enable"

4. **Create API Key**:
   - Go to: APIs & Services → Credentials
   - Click "+ CREATE CREDENTIALS" → "API Key"
   - Copy the API key

5. **Restrict API Key** (Optional but recommended):
   - Click on your API key
   - Under "API restrictions", select "Restrict key"
   - Choose "YouTube Data API v3" only
   - Save

## Step 2: Add API Key to Config

Open `config.py` and find this line:

```python
YOUTUBE_API_KEY = ""  # TODO: Add your YouTube Data API v3 key here
```

Replace it with:

```python
YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"
```

## Step 3: Install Dependencies

```bash
source venv_visual/bin/activate
pip install google-api-python-client yt-dlp
```

## Step 4: Test the Integration

```bash
python youtube_integration.py
```

This will:
- Search YouTube for movie trailers
- Save metadata to `youtube_videos.json`
- Show you 50+ trailers ready to download

## API Quotas (Free Tier)

- **Daily Limit**: 10,000 units/day
- **Search Cost**: 100 units per query
- **Effective**: ~100 searches per day

For SceneIt MVP, this is plenty!

## Next Steps

After setting up your API key, you can:
1. Run the trailer database builder
2. Download and index your first 100 trailers
3. Test SceneIt with real movie scenes!
