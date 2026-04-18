"""Per-frame people-count accuracy metrics for the dance pipeline.

The 3D output of the full pipeline contains exactly one SMPL-X track per
tracker output id; the user-visible "number of people in the 3D scene
at frame f" therefore equals the number of postprocessed tracks active
at frame f. This module measures how closely that count matches the
ground-truth count (per frame and over the whole clip).

Three flavours of "expected" count are supported:
  - From a MOT GT file: ``expected_from_gt(gt_path, num_frames)``.
  - Constant (e.g. ``2pplTest``): pass ``np.full(num_frames, K)``.
  - Custom (e.g. clips with planned entries/exits): caller-supplied.

Reported metrics per clip
-------------------------
  count_exact_acc    fraction of frames with ``pred == gt``
  count_within1_acc  fraction of frames with ``|pred - gt| <= 1``
  count_mae          mean absolute error of per-frame counts
  count_overshoot    fraction of frames with ``pred > gt``
  count_undershoot   fraction of frames with ``pred < gt``
  unique_ids_pred    total distinct postprocessed track ids
  unique_ids_gt      total distinct ids in the GT
  unique_id_ratio    pred / gt (1.0 = ideal)
  mean_pred_per_frame, mean_gt_per_frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from tracking.postprocess import Track


def expected_from_gt(gt_path: Path, num_frames: int) -> np.ndarray:
    """Read a MOT GT file and return per-frame active count (length=num_frames).

    The GT file is 1-based (frame ids start at 1). Frames with no GT row
    get a count of 0. Output length is exactly ``num_frames`` (truncated
    or zero-padded as needed).
    """
    counts = np.zeros(num_frames, dtype=np.int32)
    seen_ids: set[int] = set()
    if not gt_path.is_file():
        return counts
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            frame_id = int(float(parts[0]))
            tid = int(float(parts[1]))
        except ValueError:
            continue
        # mot15-2D + py-motmetrics min_confidence=1 filter: include row only
        # if conf == 1 (column 7, 1-indexed col 8 in MOT-Challenge layout).
        if len(parts) >= 9:
            try:
                conf = float(parts[8])  # confidence column
                if conf < 0.5:
                    continue
            except ValueError:
                pass
        f0 = frame_id - 1
        if 0 <= f0 < num_frames:
            counts[f0] += 1
            seen_ids.add(tid)
    return counts


def unique_ids_from_gt(gt_path: Path) -> int:
    """Total distinct GT track ids over the whole clip."""
    if not gt_path.is_file():
        return 0
    ids: set[int] = set()
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            ids.add(int(float(parts[1])))
        except ValueError:
            continue
    return len(ids)


def predicted_per_frame(
    tracks: Mapping[int, Track],
    num_frames: int,
) -> np.ndarray:
    """Per-frame active postprocessed-track count (length=num_frames)."""
    counts = np.zeros(num_frames, dtype=np.int32)
    for t in tracks.values():
        frames = np.asarray(t.frames, dtype=np.int64)
        frames = frames[(frames >= 0) & (frames < num_frames)]
        if frames.size == 0:
            continue
        counts[frames] += 1
    return counts


def count_metrics(
    pred_counts: np.ndarray,
    gt_counts: np.ndarray,
    *,
    pred_unique_ids: int,
    gt_unique_ids: int,
) -> Dict[str, float]:
    """Compute the per-clip count summary."""
    n = min(len(pred_counts), len(gt_counts))
    if n <= 0:
        return {
            "count_exact_acc": 0.0,
            "count_within1_acc": 0.0,
            "count_mae": 0.0,
            "count_overshoot": 0.0,
            "count_undershoot": 0.0,
            "unique_ids_pred": int(pred_unique_ids),
            "unique_ids_gt": int(gt_unique_ids),
            "unique_id_ratio": 0.0,
            "mean_pred_per_frame": 0.0,
            "mean_gt_per_frame": 0.0,
            "num_frames": 0,
        }

    p = pred_counts[:n].astype(np.int64)
    g = gt_counts[:n].astype(np.int64)
    diff = p - g
    return {
        "count_exact_acc": float(np.mean(diff == 0)),
        "count_within1_acc": float(np.mean(np.abs(diff) <= 1)),
        "count_mae": float(np.mean(np.abs(diff))),
        "count_overshoot": float(np.mean(diff > 0)),
        "count_undershoot": float(np.mean(diff < 0)),
        "unique_ids_pred": int(pred_unique_ids),
        "unique_ids_gt": int(gt_unique_ids),
        "unique_id_ratio": float(pred_unique_ids / max(gt_unique_ids, 1)),
        "mean_pred_per_frame": float(p.mean()),
        "mean_gt_per_frame": float(g.mean()),
        "num_frames": int(n),
    }


def score_tracks_counts(
    tracks: Mapping[int, Track],
    num_frames: int,
    *,
    expected_per_frame: Optional[np.ndarray] = None,
    expected_unique_ids: Optional[int] = None,
    gt_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Single entry point.

    Pass ``expected_per_frame`` + ``expected_unique_ids`` directly (used
    for clips without MOT GT, e.g. ``2pplTest``), OR pass ``gt_path``
    and the function will derive both from a MOT GT file.
    """
    if expected_per_frame is None:
        if gt_path is None:
            raise ValueError("score_tracks_counts: pass either expected_per_frame or gt_path")
        expected_per_frame = expected_from_gt(gt_path, num_frames)
        if expected_unique_ids is None:
            expected_unique_ids = unique_ids_from_gt(gt_path)
    if expected_unique_ids is None:
        expected_unique_ids = int(np.unique(expected_per_frame[expected_per_frame > 0]).size)
        # for constant counts the above is degenerate; fall back to max
        if expected_unique_ids <= 0:
            expected_unique_ids = int(expected_per_frame.max())

    pred_counts = predicted_per_frame(tracks, num_frames)
    pred_unique = len(tracks)
    return count_metrics(
        pred_counts,
        expected_per_frame,
        pred_unique_ids=pred_unique,
        gt_unique_ids=expected_unique_ids,
    )
