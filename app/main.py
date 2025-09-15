# app/main.py
import streamlit as st
import os
import sys
from PIL import Image
from datetime import datetime
import csv
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import all our analysis functions
from src.face_emotion import analyze_face
from src.text_sentiment import analyze_sentiment
from src.audio_emotion import predict_emotion as predict_voice_emotion

def save_entry(date, text, face_emotion, text_sentiment, voice_emotion):
    """Appends a new diary entry to the diary.csv file."""
    csv_path = os.path.join(project_root, "data", "diary.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Text', 'Facial Emotion', 'Text Sentiment', 'Voice Emotion'])
        writer.writerow([date, text, face_emotion, text_sentiment, voice_emotion])

# --- App Layout ---
st.title("Multi-Modal Emotion Diary")

# --- Dashboard (remains the same) ---
# ... (dashboard code is here, no changes needed)

st.markdown("---")

# --- Main App Interface for New Entry ---
st.header("How are you feeling today?")
diary_entry = st.text_area("Write down your thoughts and feelings...", height=150)

st.header("Upload a selfie to capture your expression")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Analyze My Mood"):
    if diary_entry and uploaded_image is not None:
        
        # --- Voice Recording ---
        with st.spinner("Recording your voice for 5 seconds... Please speak now."):
            sample_rate = 16000  # Sample rate for the voice model
            duration = 5
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
        
        st.success("Recording complete!")
        
        # Save the audio recording to a temporary file
        temp_audio_path = os.path.join(project_root, "data", "audio", "temp_recording.wav")
        write(temp_audio_path, sample_rate, recording)

        # --- Analysis ---
        with st.spinner("Analyzing your face, text, and voice..."):
            # Image analysis
            image = Image.open(uploaded_image).convert('RGB')
            temp_image_path = os.path.join(project_root, "data", "images", "temp_image.jpg")
            image.save(temp_image_path)
            face_result = analyze_face(temp_image_path)

            # Text analysis
            text_result = analyze_sentiment(diary_entry)
            
            # Voice analysis
            voice_result = predict_voice_emotion(temp_audio_path)

        st.success("Analysis Complete!")
        
        # --- Save the complete entry ---
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_entry(current_date, diary_entry, face_result, text_result, voice_result)
        st.success("Your entry has been saved!")
        
        # --- Display All Three Results ---
        st.subheader("Here's your multi-modal analysis:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Facial Expression")
            st.image(image, caption="Your Selfie", use_container_width=True)
            st.write(f"**Detected:** {face_result.capitalize()}")
        with col2:
            st.markdown("### Diary Sentiment")
            st.info(f'"{diary_entry}"')
            st.write(f"**Detected:** {text_result.capitalize()}")
        with col3:
            st.markdown("### Voice Tone")
            st.audio(temp_audio_path)
            st.write(f"**Detected:** {voice_result.capitalize()}")
    else:
        st.warning("Please write a diary entry and upload an image.")

# --- Diary History Section (remains the same) ---
# ... (history and chart code is here, no changes needed)