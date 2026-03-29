import json
from typing import Dict, List



def run_mock_agent(detection_summary: Dict[str, int], video_meta: Dict, ai_summary: str) -> Dict[str, List[str]]:
    """Simple rule-based agent workflow that never needs an API key."""
    total_objects = sum(detection_summary.values())
    top_class = max(detection_summary, key=detection_summary.get) if detection_summary else "none"

    insights = [
        f"Dominant object class: {top_class} ({detection_summary.get(top_class, 0)} detections).",
        f"Total detections: {total_objects} across {len(detection_summary)} categories.",
        f"Video resolution: {video_meta.get('width', 'N/A')}x{video_meta.get('height', 'N/A')} at {video_meta.get('fps', 'N/A')} fps.",
    ]

    risk_flags: List[str] = []
    if detection_summary.get("person", 0) > 20:
        risk_flags.append("High person density detected; review for possible crowd activity.")
    if detection_summary.get("knife", 0) > 0 or detection_summary.get("gun", 0) > 0:
        risk_flags.append("Potentially dangerous object detected.")
    if total_objects == 0:
        risk_flags.append("No objects detected; video may be blank, dark, or corrupted.")

    recommended_actions = [
        "Review annotated frames to verify important detections.",
        f"Inspect frames dominated by '{top_class}' for more context.",
        "Export the JSON report for downstream analysis or alerting.",
    ]

    return {
        "insights": insights,
        "risk_flags": risk_flags,
        "recommended_actions": recommended_actions,
        "summary_used": ai_summary,
    }



def run_agent(detection_summary: Dict[str, int], video_meta: Dict, ai_summary: str, mode: str = "mock") -> Dict:
    """Unified entry point. Non-mock modes safely fall back to mock."""
    _ = mode
    return run_mock_agent(detection_summary, video_meta, ai_summary)


if __name__ == "__main__":
    sample_detections = {"person": 14, "car": 6, "bicycle": 2}
    sample_meta = {"fps": 30.0, "total_frames": 900, "width": 1280, "height": 720}
    sample_summary = "The video shows a busy urban street scene."
    print(json.dumps(run_agent(sample_detections, sample_meta, sample_summary), indent=2))
