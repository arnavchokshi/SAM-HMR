"""Stage D comparison driver — writes ``metrics.json`` (plan Task 11d).

Reads the per-pipeline ``joints_world.npy`` files, harmonises them to
COCO-17 (auto-detecting SMPL-22 / MHR70 inputs), aligns frame and
dancer counts by truncation to the smaller of each, computes the
metrics from :mod:`threed.compare.metrics`, and writes a JSON summary.

The expected upstream layout per plan §4.1::

    runs/3d_compare/<clip>/
    ├── prompthmr/joints_world.npy   # (T, N, J, 3) — see Task 11f
    ├── sam_body4d/joints_world.npy  # (T, N, J, 3) — see Task 11e
    └── comparison/metrics.json      # written here

Both inputs MUST be in the same coordinate frame. Stage D-extract
(``threed/sidecar_promthmr/extract_joints.py`` for PromptHMR, the
joint-saving monkeypatch for SAM-Body4D) is responsible for that
alignment. For tripod / static-camera clips this is just the cam
frame on both sides; for moving-camera clips the PromptHMR side
projects world → cam using ``results.pkl["camera_world"]``.

Per the plan §3.6 caveat, the COCO-17 face indices (0..4) are
collapsed onto SMPL ``head`` for the PromptHMR side, so MPJPE on those
indices is artificially zero between the two pipelines. Downstream
report code can mask them out using ``threed.compare.joints.COCO17_NAMES``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from threed.compare.joints import (
    COCO17_NAMES,
    mhr70_to_coco17,
    smpl22_to_coco17,
)
from threed.compare.metrics import (
    foot_skating,
    foot_skating_world_frame,
    per_joint_jitter,
    per_joint_mpjpe,
    per_joint_mpjpe_pa,
)


def auto_reduce_to_coco17(joints: np.ndarray) -> np.ndarray:
    """Reduce ``joints`` to COCO-17 by inspecting its joint axis size.

    Supports the three layouts our pipeline can produce upstream:

    - ``17`` — already COCO-17, no-op.
    - ``22`` — SMPL body subset, reduce via :func:`smpl22_to_coco17`.
    - ``70`` — MHR70 keypoints, reduce via :func:`mhr70_to_coco17`.

    Anything else is rejected — silently producing a wrong-size
    output here would corrupt every downstream metric.
    """
    j = joints.shape[-2]
    if j == 17:
        return joints
    if j == 22:
        return smpl22_to_coco17(joints)
    if j == 70:
        return mhr70_to_coco17(joints)
    raise ValueError(
        f"auto_reduce_to_coco17 does not know how to reduce joint axis size {j}; "
        f"expected 17 (COCO), 22 (SMPL), or 70 (MHR70)"
    )


def align_arrays(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate two ``(T, N, J, 3)`` arrays to common ``T`` and ``N``.

    Truncation (not padding) is intentional: a missing dancer should
    not turn into a zero entry that drags MPJPE toward 0. The joint
    axis (``J``) MUST already match — call :func:`auto_reduce_to_coco17`
    on each input first.
    """
    if a.shape[-2] != b.shape[-2]:
        raise ValueError(
            f"align_arrays requires matching joint axis, got {a.shape} vs {b.shape}; "
            f"call auto_reduce_to_coco17 on each input first"
        )
    n_frames = min(a.shape[0], b.shape[0])
    n_dancers = min(a.shape[1], b.shape[1])
    return a[:n_frames, :n_dancers], b[:n_frames, :n_dancers]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompthmr-joints", type=Path, required=True,
        help="Path to PromptHMR-Vid joints_world.npy of shape (T, N, J, 3) "
             "(J=17 preferred; 22 also accepted and auto-reduced).",
    )
    p.add_argument(
        "--body4d-joints", type=Path, required=True,
        help="Path to SAM-Body4D joints_world.npy of shape (T, N, J, 3) "
             "(J=17 preferred; 70 also accepted and auto-reduced).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write metrics.json (parent dirs created if needed).",
    )
    p.add_argument(
        "--foot-idx", type=int, default=15,
        help="COCO-17 joint index for the foot used in foot_skating "
             "(default 15 = left_ankle).",
    )
    p.add_argument(
        "--foot-threshold", type=float, default=0.05,
        help="Foot-height threshold (m) below which the foot is 'planted'.",
    )
    p.add_argument(
        "--prompthmr-world-joints", type=Path, default=None,
        help="Optional: path to PromptHMR's world-frame joints_world.npy "
             "(SMPL-22, Y-up). If supplied, the world-frame foot-skating "
             "metric is computed via per-dancer floor calibration "
             "(see metrics.foot_skating_world_frame).",
    )
    p.add_argument(
        "--world-foot-idx", type=int, default=7,
        help="Joint index for the foot in the SMPL-22 world array "
             "(default 7 = left_ankle).",
    )
    p.add_argument(
        "--world-foot-threshold", type=float, default=0.05,
        help="Per-dancer-floor threshold (m) for world-frame foot-skating.",
    )
    args = p.parse_args(argv)

    a_raw = np.load(args.prompthmr_joints)
    b_raw = np.load(args.body4d_joints)
    print(f"[run_compare] PromptHMR raw joints: {a_raw.shape}, dtype={a_raw.dtype}")
    print(f"[run_compare] SAM-Body4D raw joints: {b_raw.shape}, dtype={b_raw.dtype}")

    a = auto_reduce_to_coco17(a_raw)
    b = auto_reduce_to_coco17(b_raw)
    print(f"[run_compare] reduced to COCO-17: phmr={a.shape}, body4d={b.shape}")

    a, b = align_arrays(a, b)
    n_frames, n_dancers = a.shape[:2]
    print(f"[run_compare] aligned: T={n_frames} frames, N={n_dancers} dancers")

    metrics = {
        "schema_version": 1,
        "joint_layout": "COCO-17",
        "joint_names": COCO17_NAMES,
        "n_frames_compared": int(n_frames),
        "n_dancers_compared": int(n_dancers),
        "n_frames_phmr": int(a_raw.shape[0]),
        "n_frames_body4d": int(b_raw.shape[0]),
        "n_dancers_phmr": int(a_raw.shape[1]),
        "n_dancers_body4d": int(b_raw.shape[1]),
        "raw_joint_axis_phmr": int(a_raw.shape[-2]),
        "raw_joint_axis_body4d": int(b_raw.shape[-2]),
        "per_joint_jitter_phmr_m_per_frame": per_joint_jitter(a).tolist(),
        "per_joint_jitter_body4d_m_per_frame": per_joint_jitter(b).tolist(),
        "per_joint_mpjpe_m": per_joint_mpjpe(a, b).tolist(),
        "per_joint_mpjpe_pa_m": per_joint_mpjpe_pa(
            a, b, per_dancer=True, allow_scale=False
        ).tolist(),
        "foot_skating_phmr_m_per_frame": foot_skating(
            a, foot_idx=args.foot_idx, threshold=args.foot_threshold
        ).tolist(),
        "foot_skating_body4d_m_per_frame": foot_skating(
            b, foot_idx=args.foot_idx, threshold=args.foot_threshold
        ).tolist(),
        "foot_idx": int(args.foot_idx),
        "foot_threshold_m": float(args.foot_threshold),
        "face_indices_collapsed_to_smpl_head": [0, 1, 2, 3, 4],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, default=_json_default))
    print(f"[run_compare] wrote {args.output}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON-serialisable")


if __name__ == "__main__":
    raise SystemExit(main())
