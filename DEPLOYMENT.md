# Deployment Guide for SceneIt

Your application is containerized and ready to "Go Live"!

## 1. Prerequisites
- [Docker Installed](https://www.docker.com/products/docker-desktop/)
- A [Render.com](https://render.com) account (or similar: Railway, Fly.io, Heroku)

> **Note:** The frontend and backend are bundled together. Deploying this Docker container deploys the **entire application** (UI + Server). Open the deployed URL to see your interface.

## 2. Option A: Run Locally (Docker)
To simulate the production environment on your machine:

```bash
# 1. Build the image (this may take a few minutes as it copies videos/frames)
docker build -t sceneit-app .

# 2. Run the container
# We pass the API key environment variable securely
docker run -d -p 5002:5002 -e YOUTUBE_API_KEY=your-key sceneit:latest
```

Then open `http://localhost:5002`.

## Alternative: Deploy to Fly.io (Recommended)
Fly.io is recommended because it supports **Persistent Volumes**, which safeguards your `sceneit.db` and search index from being wiped on every restart.

1. **Install Fly CLI**: https://fly.io/docs/hands-on/install-flyctl/
2. **Login**: `fly auth login`
3. **Initialize** (if not using provided toml): `fly launch`
4. **Create Volume**:
   ```bash
   fly vol create sceneit_data -s 1
   ```
5. **Set Secrets**:
   ```bash
   fly secrets set YOUTUBE_API_KEY=your_key_here
   ```
6. **Deploy**:
   ```bash
   fly deploy
   ```

## 3. Option B: Deploy to Render (Cloud)
This is the easiest way to share it with the world.

1.  **Push your code** to GitHub.
2.  Log in to **[Render.com](https://dashboard.render.com/)**.
3.  Click **New +** -> **Web Service**.
4.  Connect your GitHub repository.
5.  **Settings**:
    *   **Runtime**: Docker
    *   **Region**: Frankfurt (or closest to you)
    *   **Instance Type**: Free (might be slow) or Starter ($7/mo recommended for AI/Video)
6.  **Environment Variables**:
    *   Key: `YOUTUBE_API_KEY`
    *   Value: *(Your actual key)*
    *   Key: `App_Settings` (if you have others)
7.  Click **Deploy Web Service**.

### Important Note on Data
Currently, your database (`sceneit.db`) and media (`output_frames/`, `temp_videos/`) are **inside** the Docker image.
*   **Pros**: Instant deployment, everything pre-indexed.
*   **Cons**: The image is large. If you restart the hosted app on Render (Free tier), any *new* data added live (like new extractions) will be lost unless you use a persistent disk (paid feature).
*   **Recommendation**: For this MVP, this is fine. For a scaled app, we would move videos to AWS S3 and the database to PostgreSQL.

## 4. Troubleshooting
- **Memory**: The AI model (`clip-vit-base`) needs RAM. If the container crashes with "OOM" (Out of Memory), you need a larger instance type.
- **Port**: Ensure the port matches the `EXPOSE 5002` setting (Render detects this automatically usually, or set `PORT=5002`).
