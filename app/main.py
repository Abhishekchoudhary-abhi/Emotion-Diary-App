# app/main.py
import streamlit as st
import os
import sys
import csv
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------------
# PATH FIX
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# -----------------------------
# IMPORT ANALYSIS FUNCTIONS
# -----------------------------
from src.face_emotion import analyze_face
from src.text_sentiment import analyze_sentiment
from src.audio_emotion import predict_emotion as predict_voice_emotion

# -----------------------------
# SAVE ENTRY TO CSV
# -----------------------------
def save_entry(date, text, face_emotion, text_sentiment, voice_emotion):
    csv_path = os.path.join(project_root, "data", "diary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Date", "Text", "Facial Emotion", "Text Sentiment", "Voice Emotion"]
            )
        writer.writerow([date, text, face_emotion, text_sentiment, voice_emotion])

# -----------------------------
# STREAMLIT APP UI
# -----------------------------
st.title("📔 Multi-Modal Emotion Diary")
st.markdown("---")

# --- TEXT ENTRY ---
st.header("How are you feeling today?")
diary_entry = st.text_area("Write down your thoughts and feelings...", height=150)

# -----------------------------
# VIDEO SNAPSHOT (Single Frame)
# -----------------------------
st.header("📸 Capture your expression")

class SnapshotTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured = False
        self.frame = None

    def transform(self, frame):
        # Capture only if not already captured
        if not self.captured:
            self.frame = frame.to_ndarray(format="bgr24")
        return frame.to_ndarray(format="bgr24")  # show live preview

ctx = webrtc_streamer(
    key="snapshot",
    video_transformer_factory=SnapshotTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["turn:openrelay.metered.ca:80"],
             "username": "openrelayproject",
             "credential": "openrelayproject"}
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_transformer and ctx.video_transformer.frame is not None:
    if st.button("📷 Take Snapshot"):
        # Mark as captured so video stops updating
        ctx.video_transformer.captured = True

        img_rgb = cv2.cvtColor(ctx.video_transformer.frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        st.session_state["snapshot_bytes"] = buffer.getvalue()
        st.image(pil_img, caption="Your Snapshot", use_container_width=True)
        st.success("✅ Snapshot captured!")

# -----------------------------
# AUDIO UPLOAD
# -----------------------------
st.header("🎤 Upload your audio")
uploaded_audio = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])
if uploaded_audio:
    st.audio(uploaded_audio)

# -----------------------------
# ANALYZE & SAVE
# -----------------------------
if st.button("🔍 Analyze My Mood"):
    snapshot_bytes = st.session_state.get("snapshot_bytes")
    if diary_entry and snapshot_bytes and uploaded_audio is not None:
        with st.spinner("Analyzing..."):
            try:
                # --- Convert snapshot bytes to numpy array for analysis ---
                pil_img = Image.open(BytesIO(snapshot_bytes))
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # --- Face analysis ---
                temp_image_path = os.path.join(project_root, "data", "images", "temp.jpg")
                os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                pil_img.save(temp_image_path)
                face_result = analyze_face(temp_image_path)

                # --- Text analysis ---
                text_result = analyze_sentiment(diary_entry)

                # --- Audio analysis ---
                temp_audio_path = os.path.join(project_root, "data", "audio", "temp.wav")
                os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                voice_result = predict_voice_emotion(temp_audio_path)

                # Save diary entry
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_entry(current_date, diary_entry, face_result, text_result, voice_result)

                st.success("✅ Analysis Complete & Entry Saved!")

                # Results display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Facial Expression")
                    st.image(pil_img, caption="Your Snapshot", use_container_width=True)
                    st.write(f"**Detected:** {face_result.capitalize()}")
                with col2:
                    st.markdown("### Diary Sentiment")
                    st.info(f'"{diary_entry}"')
                    st.write(f"**Detected:** {text_result.capitalize()}")
                with col3:
                    st.markdown("### Voice Tone")
                    st.audio(temp_audio_path)
                    st.write(f"**Detected:** {voice_result.capitalize()}")

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.warning("⚠️ Please provide text, take a snapshot, and upload audio before analysis.")
