# SceneIt - AI Video Search Engine
# Multi-stage Docker build for production

FROM python:3.10-slim-bookworm as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (lightweight)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p output_frames temp_videos

# Expose port
EXPOSE 5002

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5002/')" || exit 1

# Run the application
# Copy entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Apply Entrypoint
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "app.py"]
