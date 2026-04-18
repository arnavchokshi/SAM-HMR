from __future__ import annotations
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np


@dataclass
class TrackEntry:
    track_id: int
    frames: np.ndarray   # (T,) int64
    bboxes: np.ndarray   # (T, 4) float32 xyxy
    confs:  np.ndarray   # (T,) float32
    masks:  Optional[np.ndarray] = None  # (T, H, W) bool — set by Stage B
    detected: Optional[np.ndarray] = None  # (T,) bool — True for real frames


def save_tracks(tracks: Dict[int, TrackEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        tid: {
            "track_id": t.track_id,
            "frames": t.frames,
            "bboxes": t.bboxes,
            "confs": t.confs,
            "masks": t.masks,
            "detected": t.detected,
        }
        for tid, t in tracks.items()
    }
    joblib.dump(payload, path)


def load_tracks(path: Path) -> Dict[int, TrackEntry]:
    payload = joblib.load(path)
    return {
        int(tid): TrackEntry(
            track_id=int(d["track_id"]),
            frames=np.asarray(d["frames"], dtype=np.int64),
            bboxes=np.asarray(d["bboxes"], dtype=np.float32),
            confs=np.asarray(d["confs"], dtype=np.float32),
            masks=(np.asarray(d["masks"], dtype=bool) if d.get("masks") is not None else None),
            detected=(np.asarray(d["detected"], dtype=bool) if d.get("detected") is not None else None),
        )
        for tid, d in payload.items()
    }
