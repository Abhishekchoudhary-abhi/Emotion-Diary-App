# app/main.py
import streamlit as st
import os
import sys
from PIL import Image
from datetime import datetime
import csv
import pandas as pd
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import all our analysis functions
from src.face_emotion import analyze_face
from src.text_sentiment import analyze_sentiment
from src.audio_emotion import predict_emotion as predict_voice_emotion

def save_entry(date, text, face_emotion, text_sentiment, voice_emotion):
    # (Saving function remains the same)
    csv_path = os.path.join(project_root, "data", "diary.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Text', 'Facial Emotion', 'Text Sentiment', 'Voice Emotion'])
        writer.writerow([date, text, face_emotion, text_sentiment, voice_emotion])

# --- App Layout ---
st.title("Multi-Modal Emotion Diary")
# (Dashboard code can remain here)
st.markdown("---")

st.header("How are you feeling today?")
diary_entry = st.text_area("Write down your thoughts and feelings...", height=150)

# --- WEBCAM SETUP ---
st.header("Capture your expression")
webrtc_ctx = webrtc_streamer(
    key="snapshot",
    video_frame_callback=lambda frame: frame,
)

if webrtc_ctx.video_receiver:
    if st.button("Take Snapshot"):
        try:
            video_frame = webrtc_ctx.video_receiver.get_latest_frame()
            
            # --- THIS IS THE FIX ---
            # Convert the raw video frame (ndarray) to a PIL Image
            img = video_frame.to_image()
            
            st.session_state["snapshot"] = img
            st.success("Snapshot taken!")
        except Exception as e:
            st.warning(f"No frame received. Please try again. Error: {e}")

if "snapshot" in st.session_state and st.session_state["snapshot"] is not None:
    st.image(st.session_state["snapshot"], caption="Your Snapshot")


st.header("Upload your audio")
uploaded_audio = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if st.button("Analyze My Mood"):
    # (Analysis button logic remains the same)
    if diary_entry and "snapshot" in st.session_state and st.session_state["snapshot"] is not None and uploaded_audio is not None:
        with st.spinner("Analyzing..."):
            image = st.session_state["snapshot"].convert('RGB')
            temp_image_path = os.path.join(project_root, "data", "images", "temp_image.jpg")
            image.save(temp_image_path)
            face_result = analyze_face(temp_image_path)

            text_result = analyze_sentiment(diary_entry)
            temp_audio_path = os.path.join(project_root, "data", "audio", "temp_recording.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            voice_result = predict_voice_emotion(temp_audio_path)

        st.success("Analysis Complete!")