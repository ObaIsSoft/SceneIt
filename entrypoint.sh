#!/bin/bash
set -e

echo "--- Entrypoint Script ---"

# 1. Handle Persistent Data Migration
# If the volume is mounted at /data, we want to populate it with our baked-in data
# ONLY if it's currently empty or missing the DB.

DATA_DIR="/data"
APP_DIR="/app"

if [ -d "$DATA_DIR" ]; then
    echo "Persistent volume detected at $DATA_DIR"
    
    # Check if DB exists in volume
    if [ ! -f "$DATA_DIR/sceneit.db" ]; then
        echo "Initializing volume with baked-in data..."
        
        # Copy DB
        if [ -f "$APP_DIR/sceneit.db" ]; then
            cp "$APP_DIR/sceneit.db" "$DATA_DIR/sceneit.db"
            echo "Copied sceneit.db"
        fi
        
        # Copy Indices/Metadata
        if [ -d "$APP_DIR/output_frames" ]; then
            mkdir -p "$DATA_DIR/output_frames"
            cp -r "$APP_DIR/output_frames/"* "$DATA_DIR/output_frames/" || true
            echo "Copied output_frames content"
        fi
        
        # Copy Temp Videos (folder structure only, since files are dockerignored)
        mkdir -p "$DATA_DIR/temp_videos"
        
    else
        echo "Existing data found in volume. Skipping initialization."
    fi
else
    echo "No persistent volume. Running with ephemeral storage."
fi

# 2. Check for missing videos and download them?
# For MVP, we might run a background check or just let them missing.
# Let's rely on the app logic or a distinct 'download' command.
# For now, just run the app.

echo "Starting Application..."
exec "$@"
