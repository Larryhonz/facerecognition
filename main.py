import pickle
import logging
from pathlib import Path
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Real-time face recognition and attendance tracking"""
    
    def __init__(self, encodings_path: Path = Path("encodings.pkl"),
                 attendance_file: Path = Path("data/attendance.csv"),
                 tolerance: float = 0.6,
                 model: str = "hog"):
        self.encodings_path = encodings_path
        self.attendance_file = attendance_file
        self.tolerance = tolerance  # Lower = stricter matching
        self.model = model  # "hog" (faster) or "cnn" (more accurate)
        
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()
        
        # Initialize attendance CSV if it doesn't exist
        self.init_attendance_file()
    
    def load_encodings(self):
        """Load pre-computed face encodings"""
        if not self.encodings_path.exists():
            logger.error(f"Encodings file not found: {self.encodings_path}")
            logger.info("Run train.py first to create encodings")
            raise FileNotFoundError("Run train.py first")
        
        with open(self.encodings_path, "rb") as f:
            data = pickle.load(f)
        
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]
        logger.info(f"Loaded {len(self.known_encodings)} face encodings")
    
    def init_attendance_file(self):
        """Create attendance CSV if it doesn't exist"""
        if not self.attendance_file.exists():
            self.attendance_file.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(columns=["Name", "Time", "Date"])
            df.to_csv(self.attendance_file, index=False)
    
    def mark_attendance(self, name: str):
        """Mark attendance for a person (avoid duplicates within 5 minutes)"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        df = pd.read_csv(self.attendance_file)
        
        # Check if person already marked in last 5 minutes
        recent = df[
            (df["Name"] == name) & 
            (df["Date"] == date_str)
        ]
        
        if not recent.empty:
            last_time = pd.to_datetime(recent.iloc[-1]["Time"])
            time_diff = (now - last_time).total_seconds() / 60
            if time_diff < 5:
                return False  # Already marked recently
        
        # Add new attendance record
        new_record = pd.DataFrame({
            "Name": [name],
            "Time": [time_str],
            "Date": [date_str]
        })
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(self.attendance_file, index=False)
        logger.info(f"Marked attendance for {name}")
        return True
    
    def recognize_faces(self, frame: np.ndarray) -> Tuple[list, list]:
        """Detect and recognize faces in a frame"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame, 
            model=self.model
        )
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, 
            face_locations
        )
        
        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings,
                face_encoding,
                tolerance=self.tolerance
            )
            distances = face_recognition.face_distance(
                self.known_encodings,
                face_encoding
            )
            
            name = "Unknown"
            confidence = 0
            
            if distances.size > 0:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]
            
            face_names.append((name, confidence))
        
        # Scale back up
        face_locations = [(top*4, right*4, bottom*4, left*4) 
                          for (top, right, bottom, left) in face_locations]
        
        return face_locations, face_names
    
    def run(self, video_source: int = 0):
        """Run real-time face recognition"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        process_every_n_frames = 2  # Process every 2nd frame for speed
        
        logger.info("Starting face recognition (Press 'q' to quit)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every nth frame
                if frame_count % process_every_n_frames == 0:
                    face_locations, face_names = self.recognize_faces(frame)
                    
                    # Draw results
                    for (top, right, bottom, left), (name, confidence) in zip(
                        face_locations, face_names
                    ):
                        # Color based on recognition
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        
                        # Draw box
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label
                        label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
                        cv2.rectangle(
                            frame,
                            (left, bottom - 35),
                            (right, bottom),
                            color,
                            cv2.FILLED
                        )
                        cv2.putText(
                            frame,
                            label,
                            (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.6,
                            (255, 255, 255),
                            1
                        )
                        
                        # Mark attendance
                        if name != "Unknown":
                            self.mark_attendance(name)
                
                # Display FPS
                cv2.putText(
                    frame,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Face Recognition - Attendance System", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released")

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run()