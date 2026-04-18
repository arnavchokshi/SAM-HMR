from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict
import numpy as np

from prune_tracks import FrameDetections, prune_detections
from threed.io import TrackEntry


def extract_tracks_from_cache(
    cache_dir: Path,
    *,
    min_total_frames: int = 60,
    min_conf: float = 0.38,
) -> Dict[int, TrackEntry]:
    """Read the FrameDetections cache produced by run_winner_stack_demo
    and return a per-track dict in the shape PromptHMR-Vid wants."""
    pkls = sorted(cache_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No FrameDetections pickle in {cache_dir}")
    if len(pkls) > 1:
        raise ValueError(f"Multiple pickles in {cache_dir}; expected exactly one")
    with open(pkls[0], "rb") as f:
        fds = pickle.load(f)

    fds = prune_detections(
        fds,
        min_total_frames=min_total_frames,
        min_conf=min_conf,
    )

    rows: Dict[int, dict] = {}
    for frame_idx, fd in enumerate(fds):
        if len(fd.tids) == 0:
            continue
        for k in range(len(fd.tids)):
            tid = int(fd.tids[k])
            d = rows.setdefault(tid, {"frames": [], "bboxes": [], "confs": []})
            d["frames"].append(frame_idx)
            d["bboxes"].append(fd.xyxys[k].tolist())
            d["confs"].append(float(fd.confs[k]))

    out: Dict[int, TrackEntry] = {}
    for tid, d in rows.items():
        out[tid] = TrackEntry(
            track_id=tid,
            frames=np.asarray(d["frames"], dtype=np.int64),
            bboxes=np.asarray(d["bboxes"], dtype=np.float32),
            confs=np.asarray(d["confs"], dtype=np.float32),
        )
    return out
