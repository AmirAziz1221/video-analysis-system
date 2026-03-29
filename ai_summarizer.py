import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()



def _duration_seconds(video_meta: Dict) -> float:
    fps = float(video_meta.get("fps", 0) or 0)
    total_frames = float(video_meta.get("total_frames", 0) or 0)
    return total_frames / fps if fps > 0 else 0.0



def _build_prompt(detection_summary: Dict[str, int], video_meta: Dict) -> str:
    duration_sec = _duration_seconds(video_meta)
    objects_str = ", ".join(f"{k} ({v} detections)" for k, v in detection_summary.items()) or "none"
    return (
        "You are a video analysis assistant.\n"
        f"A {duration_sec:.1f}-second video was analyzed using computer vision.\n"
        f"Detected objects across frames: {objects_str}.\n\n"
        "Write a clear, concise paragraph of 3 to 5 sentences describing:\n"
        "1. What the video likely shows\n"
        "2. The main subjects or objects present\n"
        "3. Any notable patterns or activity inferred from the detections\n\n"
        "Be factual and do not invent details beyond the detections."
    )



def summarize_with_openai(detection_summary: Dict[str, int], video_meta: Dict) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _build_prompt(detection_summary, video_meta)}],
            temperature=0.4,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return mock_summarize(detection_summary, video_meta)



def summarize_with_groq(detection_summary: Dict[str, int], video_meta: Dict) -> str:
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": _build_prompt(detection_summary, video_meta)}],
            temperature=0.4,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return mock_summarize(detection_summary, video_meta)



def summarize_with_gemini(detection_summary: Dict[str, int], video_meta: Dict) -> str:
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(_build_prompt(detection_summary, video_meta))
        return (response.text or "").strip()
    except Exception:
        return mock_summarize(detection_summary, video_meta)



def mock_summarize(detection_summary: Dict[str, int], video_meta: Dict) -> str:
    if not detection_summary:
        return "No objects were detected in this video."

    top_objects = list(detection_summary.items())[:3]
    top_str = ", ".join(f"{name} ({count}x)" for name, count in top_objects)
    duration_sec = _duration_seconds(video_meta)
    total_detections = sum(detection_summary.values())

    return (
        f"This {duration_sec:.1f}-second video contains activity involving {top_str}. "
        f"A total of {total_detections} object detections were recorded across "
        f"{len(detection_summary)} categories. The most frequent object was "
        f"'{top_objects[0][0]}', appearing {top_objects[0][1]} times."
    )



def get_summary(detection_summary: Dict[str, int], video_meta: Dict, provider: str = "mock") -> str:
    provider = (provider or "mock").lower().strip()
    if provider == "openai":
        return summarize_with_openai(detection_summary, video_meta)
    if provider == "groq":
        return summarize_with_groq(detection_summary, video_meta)
    if provider == "gemini":
        return summarize_with_gemini(detection_summary, video_meta)
    return mock_summarize(detection_summary, video_meta)
