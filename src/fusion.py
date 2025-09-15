# src/fusion.py
import os
from face_emotion import analyze_face
from text_sentiment import analyze_sentiment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def get_combined_analysis(image_path, diary_text):
    """
    Performs both face and text analysis and returns the combined results.
    """
    print("--- Starting Multi-Modal Analysis ---")
    print(f"Analyzing image: {os.path.basename(image_path)}")
    face_result = analyze_face(image_path)
    print(f"Facial Emotion Detected: {face_result}")
    print("-" * 20)
    print(f"Analyzing text: '{diary_text}'")
    text_result = analyze_sentiment(diary_text)
    print(f"Text Sentiment Detected: {text_result}")
    print("--- Analysis Complete ---")
    
    return {"face_emotion": face_result, "text_sentiment": text_result}

if __name__ == "__main__":
    image_file = os.path.join(PROJECT_ROOT, "data", "images", "sample_image.jpg")
    diary_entry = "I had a great day today, feeling really happy and accomplished!"

    final_results = get_combined_analysis(image_file, diary_entry)
    print("\n--- FINAL FUSION RESULT ---")
    print(final_results)