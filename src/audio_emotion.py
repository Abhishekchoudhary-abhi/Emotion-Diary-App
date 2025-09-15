# src/audio_emotion.py
import sounddevice as sd
from scipy.io.wavfile import write
import os
from transformers import pipeline

# --- Define Project Root ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def record_audio(duration=5, sample_rate=16000): # Changed sample rate for the new model
    """Records audio and saves it as a WAV file."""
    save_path = os.path.join(PROJECT_ROOT, "data", "audio", "test_recording.wav")
    print(f"Recording for {duration} seconds...")
    # This model expects a 16000 sample rate
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    write(save_path, sample_rate, recording)
    print(f"✅ Audio saved to: {save_path}")
    return save_path

def predict_emotion(file_path):
    """
    Predicts the emotion from an audio file using a Hugging Face model.
    """
    try:
        # Load the pre-trained pipeline from Hugging Face.
        # This will automatically download the model the first time it's run.
        print("Loading audio classification model from Hugging Face...")
        emotion_classifier = pipeline(
            "audio-classification", 
            model="superb/wav2vec2-base-superb-er"
        )
        print("Model loaded.")
        
        # Analyze the audio file
        results = emotion_classifier(file_path)
        
        # Find the emotion with the highest score
        highest_score = 0
        predicted_emotion = "unknown"
        for result in results:
            if result['score'] > highest_score:
                highest_score = result['score']
                predicted_emotion = result['label']
                
        return predicted_emotion
        
    except Exception as e:
        print(f"Error during emotion prediction: {e}")
        return "unknown"

if __name__ == "__main__":
    # 1. Record audio from the user
    audio_file = record_audio()
    
    # 2. Predict the emotion from the recorded audio
    print("\nAnalyzing emotion...")
    predicted_emotion = predict_emotion(audio_file)
    print(f"🎉 Predicted Emotion: {predicted_emotion}")