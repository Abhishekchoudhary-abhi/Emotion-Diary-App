import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Webcam Test")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return self.frame

webrtc_streamer(
    key="test",
    video_transformer_factory=VideoTransformer,
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
