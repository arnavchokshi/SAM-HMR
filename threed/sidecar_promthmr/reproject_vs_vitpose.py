"""Followup #4 — reproject PromptHMR-Vid 3-D joints and compare against
the bundled per-frame ViTPose 2-D detections.

PromptHMR-Vid is trained to fit ViTPose, so the reprojection of its
own ``joints_coco17_cam.npy`` (after our :mod:`project_joints` step)
should be near-zero pixel error against ``results.pkl["people"][tid]
["vitpose"]``. This script writes a per-clip ``reproj_metrics.json``
sitting next to ``metrics.json`` that captures the per-joint pixel
MPJPE, so the operator report can call out:

1. PromptHMR's 2-D consistency (low pixel error => internal sanity),
2. low-confidence ViTPose keypoints (filtered before MPJPE).

Body4D-vs-ViTPose is intentionally NOT computed here: ViTPose lives
in PromptHMR's resized image frame (e.g. 504×896 portrait for
2pplTest), but Body4D operates on ``frames_full/`` at the camera's
native resolution (e.g. 1024×576 landscape — PHMR transposes!). A
faithful Body4D-vs-ViTPose comparison requires un-rotating PHMR's
crop transform, which is non-trivial without modifying upstream
PromptHMR. Tracked as a future follow-up.

Pure-host module — joblib + numpy only. Runs in any env that can
``import joblib`` (the ``body4d`` and ``phmr_pt2.4`` envs both
qualify; the standalone test fixture skips when joblib isn't
available locally).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from threed.compare.metrics import mpjpe_2d, reproject_3d_to_2d


def load_vitpose_padded(
    people: Dict,
    n_frames: int,
    n_dancers: int,
    sorted_tids: Sequence,
) -> np.ndarray:
    """Pack per-track ViTPose into a NaN-padded ``(n_frames, n_dancers, 17, 3)``.

    The PromptHMR ``results.pkl["people"]`` dict is keyed by track-id
    and each entry holds a *subset* of frames. We materialise a dense
    ``(T, N, 17, 3)`` tensor — last axis is ``(u, v, conf)`` — that
    matches the shape and ordering of ``joints_coco17_cam.npy`` so a
    pixel-MPJPE can be computed in one numpy call.

    Slot ordering follows ``sorted_tids`` (matches our convention used
    elsewhere in the repo — see :func:`threed.sidecar_body4d.wrapper.consolidate_joints_npy`).
    Out-of-range frame indices (negative, or ``>= n_frames`` because
    of the Stage A ``--max-frames`` cap) are silently skipped.
    """
    out = np.full((n_frames, n_dancers, 17, 3), np.nan, dtype=np.float64)
    for slot, tid in enumerate(sorted_tids):
        if slot >= n_dancers:
            break
        if tid not in people:
            continue
        pp = people[tid]
        frames = np.asarray(pp["frames"], dtype=np.int64)
        vit = np.asarray(pp["vitpose"], dtype=np.float64)
        for i, f in enumerate(frames):
            f = int(f)
            if 0 <= f < n_frames:
                out[f, slot] = vit[i]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompthmr-dir", type=Path, required=True,
        help="Directory written by Stage C1 + project_joints; must contain "
             "results.pkl AND joints_coco17_cam.npy.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write reproj_metrics.json.",
    )
    p.add_argument(
        "--vitpose-conf-threshold", type=float, default=0.3,
        help="Per-keypoint ViTPose confidence threshold; below this the 2-D "
             "ground truth is masked as NaN before computing MPJPE (default 0.3).",
    )
    args = p.parse_args(argv)

    pdir: Path = args.prompthmr_dir.expanduser().resolve()
    coco_path = pdir / "joints_coco17_cam.npy"
    results_path = pdir / "results.pkl"
    if not coco_path.is_file():
        print(f"[reproj] ERROR: missing {coco_path}", file=sys.stderr)
        return 2
    if not results_path.is_file():
        print(f"[reproj] ERROR: missing {results_path}", file=sys.stderr)
        return 2

    import joblib
    results = joblib.load(results_path)

    cam = results["camera"]
    focal = float(cam["img_focal"])
    cx = float(cam["img_center"][0])
    cy = float(cam["img_center"][1])

    coco_cam = np.load(coco_path)
    n_frames, n_dancers = coco_cam.shape[:2]
    print(f"[reproj] PHMR coco17_cam {coco_cam.shape}, "
          f"focal={focal}, cx={cx}, cy={cy}")

    sorted_tids = sorted(results["people"].keys())
    print(f"[reproj] tids (sorted) = {sorted_tids}")

    vit = load_vitpose_padded(results["people"], n_frames, n_dancers, sorted_tids)
    valid_per_dancer = [
        int((~np.isnan(vit[:, d, 0, 0])).sum()) for d in range(n_dancers)
    ]
    print(f"[reproj] vitpose padded {vit.shape}, "
          f"valid frames per dancer={valid_per_dancer}")

    reproj_phmr = reproject_3d_to_2d(coco_cam, focal=focal, cx=cx, cy=cy)
    print(f"[reproj] PHMR reproj-2D {reproj_phmr.shape}")

    conf = vit[..., 2]
    vit_uv = vit[..., :2].astype(np.float64).copy()
    low_conf_mask = conf < args.vitpose_conf_threshold
    n_low = int(low_conf_mask.sum())
    vit_uv[low_conf_mask] = np.nan
    print(f"[reproj] masked {n_low} low-confidence ViTPose keypoints "
          f"(threshold={args.vitpose_conf_threshold})")

    pjpe_phmr = mpjpe_2d(reproj_phmr, vit_uv)
    pjpe_phmr_mean = float(np.nanmean(pjpe_phmr))
    print(f"[reproj] mean PHMR-vs-ViTPose pixel MPJPE = {pjpe_phmr_mean:.3f} px")

    metrics: Dict = {
        "schema_version": 1,
        "joint_layout": "COCO-17",
        "n_frames": int(n_frames),
        "n_dancers": int(n_dancers),
        "vitpose_conf_threshold": float(args.vitpose_conf_threshold),
        "n_low_confidence_keypoints": n_low,
        "phmr_focal": focal,
        "phmr_cx": cx,
        "phmr_cy": cy,
        "tids_sorted": [int(t) for t in sorted_tids],
        "valid_frames_per_dancer_phmr": valid_per_dancer,
        "per_joint_mpjpe_phmr_vs_vitpose_px": pjpe_phmr.tolist(),
        "mean_mpjpe_phmr_vs_vitpose_px": pjpe_phmr_mean,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, default=_json_default))
    print(f"[reproj] wrote {args.output}")
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
