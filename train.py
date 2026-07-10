import os
import pickle
from pathlib import Path
import face_recognition
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEncoder:
    """Handles encoding and saving face data"""
    
    def __init__(self, encodings_path: Path = Path("encodings.pkl")):
        self.encodings_path = encodings_path
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
    
    def encode_faces(self, faces_dir: Path) -> bool:
        """
        Encode all faces in the faces directory
        Expected structure: faces_dir/person_name/*.jpg
        """
        if not faces_dir.exists():
            logger.error(f"Faces directory not found: {faces_dir}")
            return False
        
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        face_count = 0
        
        # Iterate through each person's folder
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            logger.info(f"Processing faces for: {person_name}")
            
            images = [f for f in person_dir.iterdir() 
                     if f.suffix.lower() in valid_extensions]
            
            if not images:
                logger.warning(f"No images found for {person_name}")
                continue
            
            for image_path in images:
                try:
                    # Load image
                    image = face_recognition.load_image_file(str(image_path))
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if not face_encodings:
                        logger.warning(f"No face detected in {image_path.name}")
                        continue
                    
                    # Use the first face found
                    encoding = face_encodings[0]
                    self.known_encodings.append(encoding)
                    self.known_names.append(person_name)
                    face_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
        
        if face_count == 0:
            logger.error("No faces were successfully encoded")
            return False
        
        logger.info(f"Successfully encoded {face_count} faces")
        self.save_encodings()
        return True
    
    def save_encodings(self):
        """Save encodings to file"""
        data = {
            "encodings": self.known_encodings,
            "names": self.known_names
        }
        
        with open(self.encodings_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Encodings saved to {self.encodings_path}")

if __name__ == "__main__":
    faces_directory = Path("data/faces")
    encoder = FaceEncoder()
    encoder.encode_faces(faces_directory)