# 🎬 Agentic AI Video Analysis System

An end-to-end AI-powered video analysis system that: - extracts frames
from videos - detects objects using YOLOv8 - generates AI-based
summaries - produces intelligent insights using agent workflows -
provides both CLI and Streamlit web interface

------------------------------------------------------------------------

## 🚀 Features

-   📹 Supports video file, webcam, or URL input\
-   🖼️ Frame extraction with configurable intervals\
-   🤖 Object detection using YOLOv8 (Ultralytics)\
-   🧠 AI-generated summaries (OpenAI / Gemini / Groq / Mock)\
-   🧩 Agentic workflow (LangChain / CrewAI / Mock)\
-   🌐 Interactive web UI using Streamlit\
-   📊 Visualization of detections and insights\
-   📥 Export full JSON analysis report

------------------------------------------------------------------------

## 📁 Project Structure

video_analysis_system/ │ ├── app.py ├── main.py ├── video_input.py ├──
frame_extractor.py ├── object_detector.py ├── ai_summarizer.py ├──
agent_workflow.py ├── requirements.txt ├── .env ├── README.md ├──
output/ ├── extracted_frames/ └── annotated_frames/

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
python -m pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

### CLI

``` bash
python main.py sample.mp4 --ai mock
```

### Streamlit

``` bash
python -m streamlit run app.py
```

------------------------------------------------------------------------

## 🔑 Environment Variables

Create a `.env` file:

GROQ_API_KEY=your_key_here\
OPENAI_API_KEY=your_key_here\
GEMINI_API_KEY=your_key_here

------------------------------------------------------------------------

## 📊 Output

-   Extracted frames\
-   Annotated frames\
-   JSON report

------------------------------------------------------------------------

## ⚠️ Notes

-   First run downloads YOLO model
-   Use mock mode if no API keys
-   Recommended Python 3.10/3.11

------------------------------------------------------------------------

## 👨‍💻 Author

AI Video Analysis System (Agentic AI + Computer Vision)# video-analysis-system
