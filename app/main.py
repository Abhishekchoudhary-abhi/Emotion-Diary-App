# app/main.py
import streamlit as st
import os
import sys
from PIL import Image
from datetime import datetime
import csv
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import all our analysis functions
from src.face_emotion import analyze_face
from src.text_sentiment import analyze_sentiment
from src.audio_emotion import predict_emotion as predict_voice_emotion

# --- Session State to hold the snapshot ---
if 'snapshot' not in st.session_state:
    st.session_state['snapshot'] = None

# --- Webcam Frame Capturing ---
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # This function is required by the library, but we don't need to process the live feed.
        # We'll just return the frame as is.
        img = frame.to_ndarray(format="bgr24")
        return img

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

st.header("Capture your expression")
# Create the WebRTC streamer
ctx = webrtc_streamer(key="webcam", video_transformer_class=VideoTransformer)

if ctx.video_transformer:
    if st.button("Take Snapshot"):
        # Take a snapshot from the video feed
        snapshot = ctx.video_transformer.get_frame()
        if snapshot is not None:
            st.session_state['snapshot'] = Image.fromarray(snapshot)
            st.success("Snapshot taken!")
            st.image(st.session_state['snapshot'], caption="Your Snapshot")
        else:
            st.warning("Could not take snapshot. Please try again.")

st.header("Upload your audio")
uploaded_audio = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if st.button("Analyze My Mood"):
    # Check if all inputs are ready
    if diary_entry and st.session_state['snapshot'] is not None and uploaded_audio is not None:
        with st.spinner("Analyzing..."):
            # --- Image Processing (using the snapshot) ---
            image = st.session_state['snapshot'].convert('RGB')
            temp_image_path = os.path.join(project_root, "data", "images", "temp_image.jpg")
            image.save(temp_image_path)
            face_result = analyze_face(temp_image_path)

            # (Text and Audio analysis remains the same)
            text_result = analyze_sentiment(diary_entry)
            temp_audio_path = os.path.join(project_root, "data", "audio", "temp_recording.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            voice_result = predict_voice_emotion(temp_audio_path)

        st.success("Analysis Complete!")
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_entry(current_date, diary_entry, face_result, text_result, voice_result)
        st.success("Your entry has been saved!")
        
        # (Display results section remains the same)
        st.subheader("Here's your multi-modal analysis:")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown("### Facial Expression")
            st.image(image, caption="Your Snapshot", use_container_width=True)
            st.write(f"**Detected:** {face_result.capitalize()}")
        with res_col2:
            st.markdown("### Diary Sentiment")
            st.info(f'"{diary_entry}"')
            st.write(f"**Detected:** {text_result.capitalize()}")
        with res_col3:
            st.markdown("### Voice Tone")
            st.audio(temp_audio_path)
            st.write(f"**Detected:** {voice_result.capitalize()}")
    else:
        st.warning("Please provide a diary entry, take a snapshot, and upload an audio file.")

# (Diary History section remains the same)