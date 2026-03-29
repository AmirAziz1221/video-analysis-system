import argparse
import json
import os
from typing import Any, Dict, Union

from ai_summarizer import get_summary
from agent_workflow import run_agent
from frame_extractor import extract_frames, get_frame_stats
from object_detector import detect_objects_in_frames, load_detector, summarize_detections
from video_input import load_video, release_video



def run_pipeline(
    video_source: Union[str, int],
    frame_interval: int = 30,
    max_frames: int = 20,
    confidence: float = 0.4,
    ai_provider: str = "mock",
    agent_mode: str = "mock",
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Run the full video analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frames")
    annotated_dir = os.path.join(output_dir, "annotated")

    cap, meta = load_video(video_source)
    try:
        frame_paths, frame_arrays = extract_frames(
            cap,
            output_dir=frame_dir,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )
    finally:
        release_video(cap)

    stats = get_frame_stats(frame_arrays)
    model = load_detector()
    all_detections = detect_objects_in_frames(
        model,
        frame_arrays,
        frame_paths,
        confidence_threshold=confidence,
        output_dir=annotated_dir,
    )
    detection_summary = summarize_detections(all_detections)
    ai_summary = get_summary(detection_summary, meta, provider=ai_provider)
    agent_report = run_agent(detection_summary, meta, ai_summary, mode=agent_mode)

    full_report = {
        "video_source": str(video_source),
        "video_metadata": meta,
        "frame_stats": stats,
        "detection_summary": detection_summary,
        "all_detections": all_detections,
        "ai_summary": ai_summary,
        "agent_report": agent_report,
    }

    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    return full_report



def parse_source(raw_video: str) -> Union[str, int]:
    return int(raw_video) if raw_video.isdigit() else raw_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Video Analysis System")
    parser.add_argument("video", help="Path to video file, URL, or webcam index (0, 1, ...)")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum number of frames")
    parser.add_argument("--confidence", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--ai", type=str, default="mock", help="AI provider: mock|groq|openai|gemini")
    parser.add_argument("--agent", type=str, default="mock", help="Agent mode: mock")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    report = run_pipeline(
        video_source=parse_source(args.video),
        frame_interval=args.interval,
        max_frames=args.max_frames,
        confidence=args.confidence,
        ai_provider=args.ai,
        agent_mode=args.agent,
        output_dir=args.output,
    )

    print(json.dumps(report["detection_summary"], indent=2))
