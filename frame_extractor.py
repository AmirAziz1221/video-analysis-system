import os
from typing import Dict, List, Tuple

import cv2
import numpy as np



def extract_frames(
    cap: cv2.VideoCapture,
    output_dir: str = "extracted_frames",
    frame_interval: int = 30,
    max_frames: int = 50,
) -> Tuple[List[str], List[np.ndarray]]:
    """Extract frames from an opened cv2.VideoCapture object."""
    if frame_interval <= 0:
        raise ValueError("frame_interval must be greater than 0.")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than 0.")

    os.makedirs(output_dir, exist_ok=True)

    frame_paths: List[str] = []
    frame_arrays: List[np.ndarray] = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            frame_paths.append(filepath)
            frame_arrays.append(frame.copy())
            saved_count += 1

            if saved_count >= max_frames:
                break

        frame_count += 1

    return frame_paths, frame_arrays



def get_frame_stats(frame_arrays: List[np.ndarray]) -> Dict[str, int]:
    """Return basic stats about the extracted frames."""
    if not frame_arrays:
        return {"total_extracted": 0, "frame_height": 0, "frame_width": 0}

    height, width = frame_arrays[0].shape[:2]
    return {
        "total_extracted": len(frame_arrays),
        "frame_height": int(height),
        "frame_width": int(width),
    }
