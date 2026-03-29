import json
import os
import tempfile

import cv2
import streamlit as st
from PIL import Image

from ai_summarizer import get_summary
from agent_workflow import run_agent
from frame_extractor import extract_frames
from object_detector import detect_objects_in_frames, load_detector, summarize_detections
from video_input import load_video, release_video

st.set_page_config(page_title="AI Video Analyzer", page_icon="🎬", layout="wide")
st.title("🎬 AI Video Analysis System")
st.caption("Upload a video, extract frames, detect objects, and generate insights.")

with st.sidebar:
    st.header("Settings")
    frame_interval = st.slider("Frame interval", 5, 120, 30, 5)
    max_frames = st.slider("Max frames", 5, 100, 20, 5)
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    ai_provider = st.selectbox("AI provider", ["mock", "groq", "openai", "gemini"], index=0)
    st.info("Mock mode works without any API key.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is None:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded_file.read())
    video_path = tmp.name

st.success(f"Uploaded: {uploaded_file.name}")

if st.button("Analyze Video", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading video..."):
            cap, meta = load_video(video_path)

        with st.spinner("Extracting frames..."):
            tmpdir = tempfile.mkdtemp()
            frame_dir = os.path.join(tmpdir, "frames")
            annotated_dir = os.path.join(tmpdir, "annotated")
            paths, arrays = extract_frames(
                cap,
                output_dir=frame_dir,
                frame_interval=frame_interval,
                max_frames=max_frames,
            )
            release_video(cap)

        st.write(meta)
        st.success(f"Extracted {len(paths)} frames")

        if arrays:
            st.subheader("Extracted frame preview")
            cols = st.columns(min(5, len(arrays)))
            for i, (col, arr) in enumerate(zip(cols, arrays[:5])):
                col.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), caption=f"Frame {i}", use_container_width=True)

        with st.spinner("Running object detection..."):
            model = load_detector()
            all_detections = detect_objects_in_frames(
                model,
                arrays,
                paths,
                confidence_threshold=confidence,
                output_dir=annotated_dir,
            )
            detection_summary = summarize_detections(all_detections)

        st.subheader("Detection summary")
        if detection_summary:
            st.bar_chart(detection_summary)
        else:
            st.warning("No objects detected.")

        if all_detections:
            st.subheader("Annotated frame preview")
            cols = st.columns(min(5, len(all_detections)))
            for i, (col, det) in enumerate(zip(cols, all_detections[:5])):
                img = Image.open(det["annotated_path"])
                col.image(img, caption=f"Annotated {i}", use_container_width=True)

        with st.spinner("Generating summary..."):
            ai_summary = get_summary(detection_summary, meta, provider=ai_provider)
        st.subheader("AI summary")
        st.info(ai_summary)

        agent_report = run_agent(detection_summary, meta, ai_summary, mode="mock")
        st.subheader("Agent report")
        st.json(agent_report)

        report_json = json.dumps(
            {
                "video_metadata": meta,
                "detection_summary": detection_summary,
                "all_detections": all_detections,
                "ai_summary": ai_summary,
                "agent_report": agent_report,
            },
            indent=2,
        )
        st.download_button(
            "Download JSON report",
            report_json,
            file_name="video_analysis_report.json",
            mime="application/json",
            use_container_width=True,
        )
    finally:
        try:
            release_video(cap)  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            os.unlink(video_path)
        except Exception:
            pass
