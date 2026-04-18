"""MOT file I/O and py-motmetrics scoring (IoU 0.5, MOTChallenge-style)."""

from __future__ import annotations

import io
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from prune_tracks import FrameDetections


def frame_detections_to_mot_rows(
    frames: list[FrameDetections],
    *,
    frame_offset: int = 1,
) -> list[str]:
    """
    Convert per-frame detections to MOTChallenge lines.
    frame_offset: 1-based frame index for the first element of `frames` (usually 1).
    Coordinates follow motmetrics convention: X,Y are 1-based in the file (MATLAB style).
    """
    lines: list[str] = []
    for i, fd in enumerate(frames):
        frame_id = frame_offset + i
        if len(fd.tids) == 0:
            continue
        xyxys = np.asarray(fd.xyxys, dtype=np.float64)
        confs = np.asarray(fd.confs, dtype=np.float64)
        tids = np.asarray(fd.tids, dtype=np.float64)
        for row in range(len(tids)):
            x1, y1, x2, y2 = xyxys[row]
            w = float(x2 - x1)
            h = float(y2 - y1)
            conf = float(confs[row])
            tid = int(tids[row])
            # motmetrics load_motchallenge subtracts 1 from X,Y — store 1-based top-left.
            X = float(x1 + 1.0)
            Y = float(y1 + 1.0)
            lines.append(
                f"{frame_id},{tid},{X:.6f},{Y:.6f},{w:.6f},{h:.6f},{conf},1,-1"
            )
    return lines


def write_mot_txt(frames: list[FrameDetections], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = frame_detections_to_mot_rows(frames)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def load_gt_dataframe(gt_path: Path) -> pd.DataFrame:
    return mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)


def mot_metrics_for_sequence(
    gt_path: Path,
    pred_frames: list[FrameDetections],
) -> dict[str, float]:
    """Compute CLEAR + ID metrics for one sequence; IoU match threshold 0.5."""
    gt = load_gt_dataframe(gt_path)
    rows = frame_detections_to_mot_rows(pred_frames)
    buf = io.StringIO("\n".join(rows) + "\n" if rows else "")
    ts = mm.io.loadtxt(buf, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    names = [
        "mota",
        "idf1",
        "num_switches",
        "num_fragmentations",
        "num_misses",
        "num_false_positives",
        "mostly_tracked",
        "mostly_lost",
        "num_detections",
    ]
    summary = mh.compute(acc, metrics=names, name="seq")
    out: dict[str, float] = {}
    for n in names:
        try:
            v = float(summary[n].iloc[0])
        except Exception:
            v = float("nan")
        out[n] = v
    out["num_frames"] = float(len(gt.index.get_level_values(0).unique()))
    return out


def _pred_to_ts_df(pred_frames: list[FrameDetections]) -> pd.DataFrame:
    rows = frame_detections_to_mot_rows(pred_frames)
    buf = io.StringIO("\n".join(rows) + "\n" if rows else "")
    return mm.io.loadtxt(buf, fmt="mot15-2D")


def compare_accumulator(
    gt_path: Path,
    pred_frames: list[FrameDetections],
):
    """MOTAccumulator for IoU 0.5 matching."""
    gt = load_gt_dataframe(gt_path)
    ts = _pred_to_ts_df(pred_frames)
    return mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)


def overall_metrics_from_accumulators(
    accs: list,
    names: list[str],
) -> pd.DataFrame:
    """Per-sequence + OVERALL row (concatenated accumulator)."""
    mh = mm.metrics.create()
    metrics = [
        "mota",
        "idf1",
        "num_switches",
        "num_fragmentations",
        "num_misses",
        "num_false_positives",
        "mostly_tracked",
        "mostly_lost",
        "num_detections",
    ]
    return mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=True
    )
