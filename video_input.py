import os
from typing import Any, Dict, Tuple, Union

import cv2


VideoSource = Union[str, int]


def load_video(source: VideoSource) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
    """Load a video from a file path, URL, or webcam index."""
    if isinstance(source, str):
        source = source.strip()
        if not source:
            raise ValueError("Video source cannot be empty.")
        if not source.startswith(("http://", "https://")) and not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta = {
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "source": str(source),
    }
    return cap, meta



def release_video(cap: cv2.VideoCapture) -> None:
    """Safely release the video capture object."""
    if cap is not None and cap.isOpened():
        cap.release()
