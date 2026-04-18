from __future__ import annotations
from pathlib import Path
import cv2


def extract_frames(
    video: Path,
    out_dir_resized: Path,
    out_dir_full: Path,
    *,
    max_height: int = 896,
) -> int:
    """Extract frames from a video into TWO folders:
    - out_dir_resized: frames downscaled so height <= max_height (for PromptHMR)
    - out_dir_full:    frames at original resolution (for SAM-Body4D)

    Returns the number of frames written.
    """
    out_dir_resized.mkdir(parents=True, exist_ok=True)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            if h > max_height:
                scale = max_height / h
                resized = cv2.resize(frame, (int(round(w * scale)), max_height),
                                     interpolation=cv2.INTER_AREA)
            else:
                resized = frame
            cv2.imwrite(str(out_dir_resized / f"{n:08d}.jpg"), resized)
            cv2.imwrite(str(out_dir_full / f"{n:08d}.jpg"), frame)
            n += 1
    finally:
        cap.release()
    return n
