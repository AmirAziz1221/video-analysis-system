import os
from typing import Any, Dict, List

import cv2

MODEL_NAME = "yolov8n.pt"



def load_detector(model_name: str = MODEL_NAME):
    """Load the YOLOv8 model lazily so import errors happen only when needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return YOLO(model_name)



def detect_objects_in_frames(
    model: Any,
    frame_arrays: List[Any],
    frame_paths: List[str],
    confidence_threshold: float = 0.4,
    output_dir: str = "annotated_frames",
) -> List[Dict[str, Any]]:
    """Run YOLOv8 on each frame and save annotated images."""
    os.makedirs(output_dir, exist_ok=True)
    all_detections: List[Dict[str, Any]] = []

    for i, (frame, path) in enumerate(zip(frame_arrays, frame_paths)):
        results = model(frame, verbose=False)[0]
        frame_detections: List[Dict[str, Any]] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = str(model.names[cls_id])
            bbox = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            frame_detections.append(
                {
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": bbox,
                }
            )

        annotated = results.plot()
        out_name = f"annotated_{i:04d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, annotated)

        all_detections.append(
            {
                "frame_index": i,
                "frame_path": path,
                "annotated_path": out_path,
                "objects": frame_detections,
            }
        )

    return all_detections



def summarize_detections(all_detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count how often each class appears across all frames."""
    counts: Dict[str, int] = {}
    for frame in all_detections:
        for obj in frame.get("objects", []):
            cls_name = obj["class"]
            counts[cls_name] = counts.get(cls_name, 0) + 1

    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
