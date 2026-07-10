#!/usr/bin/env python3
"""
Helper script to download sample face images for training.
Uses the LFW (Labeled Faces in the Wild) dataset as a source.
"""

import os
import urllib.request
from pathlib import Path

# Sample face image URLs (using public domain/CC licensed images)
FACE_URLS = [
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama1.jpg",
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
]

def download_sample_images():
    """Download sample face images for testing"""
    data_dir = Path("data/faces/john")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, url in enumerate(FACE_URLS, 1):
        try:
            filename = data_dir / f"image{idx}.jpg"
            if filename.exists():
                print(f"✓ {filename} already exists")
                continue
            
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"✗ Error downloading {url}: {e}")
    
    print(f"\nData structure created:")
    print(f"data/faces/john/")
    for img in data_dir.glob("*.jpg"):
        print(f"  └── {img.name}")

if __name__ == "__main__":
    download_sample_images()
