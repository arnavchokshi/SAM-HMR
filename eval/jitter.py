"""Per-track stability / jitter metrics.

The downstream pipeline (SAM 2.1 masks → VitPose-H keypoints → PromptHMR
→ world alignment) is driven by the per-frame bounding box for each
track. If a box CENTER, WIDTH, or HEIGHT jitters frame-to-frame, the
SMPL-X body inherits that jitter as visible body-shape and position
wobble in the final 3D output. So the "accuracy" of the tracker isn't
just count match -- it's also *how stable* each track's box geometry is.

This module exposes three families of metrics, computed per track and
then aggregated per clip:

  * **Center jitter**: short-window standard deviation of (cx, cy)
    velocity, measured in pixels/frame.
  * **Size jitter**: short-window standard deviation of width and
    height, measured in pixels.
  * **Aspect-ratio jump**: the largest single-frame jump in (w/h),
    measured as a ratio. Big jumps here directly cause SMPL-X body
    proportions to flicker.

For each metric we report MEAN, MEDIAN, and P95 across the per-track
distribution -- the P95 is the most useful for the user's "no crazy
changes in dimensions in super short time" requirement, because a
single bad track can cause visible 3D artifacts even if mean is fine.

All metrics are computed AFTER smoothing/post-processing, i.e. they
measure the *output* the 3D layer will see, not the raw YOLO/BoT-SORT
boxes.
"""
from __future__ import annotations
from typing import Dict, Iterable, Mapping
import numpy as np

from tracking.postprocess import Track


def _xyxy_to_cwh(box: np.ndarray) -> np.ndarray:
    """Convert (x1,y1,x2,y2) -> (cx, cy, w, h) elementwise on the last dim."""
    box = np.asarray(box, dtype=np.float32)
    cx = 0.5 * (box[..., 0] + box[..., 2])
    cy = 0.5 * (box[..., 1] + box[..., 3])
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def per_track_jitter(track: Track) -> Dict[str, float]:
    """Compute per-track jitter / stability metrics.

    Returns a dict with:
      * ``n_frames``: track length
      * ``center_vel_std_px``: std of frame-to-frame center velocity (||v||)
      * ``width_diff_std_px``: std of |Δw| between consecutive frames
      * ``height_diff_std_px``: std of |Δh| between consecutive frames
      * ``aspect_max_jump``: max ratio of consecutive aspect ratios,
        normalised so 1.0 = no change (always >= 1.0)
      * ``size_outlier_frac``: fraction of frames where |Δw|/median_w
        OR |Δh|/median_h exceeds 0.20 (visually noticeable size pop)
      * ``center_outlier_frac``: fraction of frames where center moves
        more than ``0.5 * median_size`` in a single frame (a "teleport")
    """
    if len(track.frames) < 3:
        return {
            "n_frames": float(len(track.frames)),
            "center_vel_std_px": 0.0,
            "width_diff_std_px": 0.0,
            "height_diff_std_px": 0.0,
            "aspect_max_jump": 1.0,
            "size_outlier_frac": 0.0,
            "center_outlier_frac": 0.0,
        }
    cwh = _xyxy_to_cwh(track.bboxes)
    cx, cy, w, h = cwh[:, 0], cwh[:, 1], cwh[:, 2], cwh[:, 3]
    dx = np.diff(cx); dy = np.diff(cy)
    dw = np.abs(np.diff(w)); dh = np.abs(np.diff(h))
    vel = np.sqrt(dx * dx + dy * dy)
    aspect = w / np.maximum(h, 1e-6)
    aspect_ratio = aspect[1:] / np.maximum(aspect[:-1], 1e-6)
    aspect_jump = np.maximum(aspect_ratio, 1.0 / np.maximum(aspect_ratio, 1e-6))
    median_w = float(np.median(w))
    median_h = float(np.median(h))
    median_size = max(0.5 * (median_w + median_h), 1.0)
    size_outlier = ((dw / max(median_w, 1.0)) > 0.20) | ((dh / max(median_h, 1.0)) > 0.20)
    center_outlier = vel > (0.5 * median_size)
    return {
        "n_frames": float(len(track.frames)),
        "center_vel_std_px": float(np.std(vel)),
        "width_diff_std_px": float(np.std(dw)),
        "height_diff_std_px": float(np.std(dh)),
        "aspect_max_jump": float(np.max(aspect_jump)) if aspect_jump.size else 1.0,
        "size_outlier_frac": float(np.mean(size_outlier)),
        "center_outlier_frac": float(np.mean(center_outlier)),
    }


def clip_jitter(tracks: Mapping[int, Track]) -> Dict[str, float]:
    """Aggregate per-track jitter into per-clip statistics.

    For each per-track scalar we report:
      * ``<metric>_mean``: arithmetic mean across all tracks
      * ``<metric>_p95``: 95th-percentile across all tracks (the worst
        track effectively dictates 3D quality)
      * ``<metric>_max``: literal worst track (sanity check / debugging)
    """
    if not tracks:
        return {}
    keys = [
        "center_vel_std_px",
        "width_diff_std_px",
        "height_diff_std_px",
        "aspect_max_jump",
        "size_outlier_frac",
        "center_outlier_frac",
    ]
    accum: Dict[str, list[float]] = {k: [] for k in keys}
    for t in tracks.values():
        per = per_track_jitter(t)
        for k in keys:
            accum[k].append(per[k])
    out: Dict[str, float] = {"jitter_n_tracks": float(len(tracks))}
    for k in keys:
        arr = np.asarray(accum[k], dtype=np.float64)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_p95"] = float(np.percentile(arr, 95))
        out[f"{k}_max"] = float(arr.max())
    return out


def fmt_jitter_summary(stats: Mapping[str, float]) -> str:
    """One-line human-readable jitter summary."""
    if not stats:
        return "(no tracks)"
    return (
        f"center_vel_std mean/p95={stats['center_vel_std_px_mean']:.2f}/"
        f"{stats['center_vel_std_px_p95']:.2f}px  "
        f"size_diff_std mean/p95={(stats['width_diff_std_px_mean'] + stats['height_diff_std_px_mean'])*0.5:.2f}/"
        f"{(stats['width_diff_std_px_p95'] + stats['height_diff_std_px_p95'])*0.5:.2f}px  "
        f"size_outlier mean/p95={stats['size_outlier_frac_mean']:.3f}/{stats['size_outlier_frac_p95']:.3f}  "
        f"center_outlier mean/p95={stats['center_outlier_frac_mean']:.3f}/{stats['center_outlier_frac_p95']:.3f}  "
        f"aspect_max_jump mean/p95={stats['aspect_max_jump_mean']:.3f}/{stats['aspect_max_jump_p95']:.3f}"
    )
