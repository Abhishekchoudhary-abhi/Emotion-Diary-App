# src/face_emotion.py
import cv2
from deepface import DeepFace
import os

def analyze_face(img_path):
    """
    Analyzes emotions from a given image file path.
    Returns the dominant emotion.
    """
    try:
        # Using a more accurate face detector backend
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='retinaface'
        )
        dominant_emotion = result[0]['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print(f"Error in face analysis: {e}")
        return "unknown"

# This block is for testing the script directly
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    IMAGE_NAME = "sample_image.jpg" 
    img_path = os.path.join(project_root, "data", "images", IMAGE_NAME)

    if os.path.exists(img_path):
        emotion = analyze_face(img_path)
        print(f"Detected Emotion: {emotion}")
    else:
        print(f"Test image not found at: {img_path}")