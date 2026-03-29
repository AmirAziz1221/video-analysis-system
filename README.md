# 🎬 Agentic AI Video Analysis System

## 📌 Overview

This project is an end-to-end AI-powered video analysis system that: -
Extracts frames from videos - Detects objects using YOLOv8 - Generates
AI summaries - Produces intelligent insights using agent workflows -
Provides both CLI and Streamlit web interface

------------------------------------------------------------------------

## 🚀 Features

-   📹 Supports video file, webcam, or URL input\
-   🖼️ Frame extraction with configurable intervals\
-   🤖 Object detection using YOLOv8\
-   🧠 AI-generated summaries (Mock / OpenAI / Gemini / Groq)\
-   🧩 Agentic workflow (Mock / LangChain / CrewAI)\
-   🌐 Interactive UI using Streamlit\
-   📊 Visualization of results\
-   📥 Export JSON report

------------------------------------------------------------------------

## 📁 Project Structure

    video_analysis_system/
    │
    ├── app.py
    ├── main.py
    ├── video_input.py
    ├── frame_extractor.py
    ├── object_detector.py
    ├── ai_summarizer.py
    ├── agent_workflow.py
    ├── requirements.txt
    ├── README.md
    │
    ├── output/
    ├── extracted_frames/
    └── annotated_frames/

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
python -m pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

### Run CLI

``` bash
python main.py sample.mp4 --ai mock
```

### Run Web App

``` bash
python -m streamlit run app.py
```

------------------------------------------------------------------------

## 🔑 Environment Variables

Create a `.env` file:

    GROQ_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here
    GEMINI_API_KEY=your_key_here

------------------------------------------------------------------------

## 📊 Output

-   Extracted frames\
-   Annotated frames\
-   JSON report

------------------------------------------------------------------------

## ⚠️ Notes

-   First run downloads YOLO model automatically\
-   Use `mock` mode if no API key\
-   Recommended Python 3.10 or 3.11

------------------------------------------------------------------------

## 👨‍💻 Author

AI Video Analysis System Project
