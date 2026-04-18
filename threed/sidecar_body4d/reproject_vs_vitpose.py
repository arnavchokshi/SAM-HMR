"""Followup #4 (Body4D side) — reproject SAM-Body4D's per-frame 3-D
joints into the original frame's pixel space and compare against the
ViTPose 17-keypoint detections that PromptHMR-Vid bundles into its
``results.pkl``.

This is the cross-pipeline accuracy metric ("how well does each
pipeline put the dancers where the camera sees them?") and is what
matches what an operator sees on the side-by-side video. It complements
:mod:`threed.sidecar_promthmr.reproject_vs_vitpose`, which is a
self-consistency check on PHMR alone.

Coordinate-system bookkeeping (this is the part that bit us the first
time):

- Body4D's ``joints_world.npy`` holds per-(frame, dancer) MHR70 joints
  in *each dancer's own canonical frame* (X∈[-0.5, 0.5], Y∈[-1.4, 0.0],
  Z∈[-0.6, 0.0] roughly). To get the camera frame, we add the
  per-(frame, dancer) translation ``cam_t`` from
  ``focal_4d_individual/<pid>/<frame>.json``.
- Body4D's per-(frame, dancer) ``focal_length`` is in *native* pixel
  space; the principal point is the native frame center (W/2, H/2).
- PromptHMR-Vid runs on a *resized* canvas (e.g. 504×896 portrait for
  2pplTest), so its ``vitpose`` field — even though the keypoints are
  in pixel coords — is in PHMR's canvas, not native. We rescale by
  the canvas-vs-native ratio before comparing against Body4D.
- The native frame size is read from
  ``intermediates/frames_full/00000000.jpg``.

Output: extends the per-clip ``comparison/reproj_metrics.json`` written
by the PHMR-side script with ``mean_mpjpe_body4d_vs_vitpose_px``,
``per_joint_mpjpe_body4d_vs_vitpose_px`` (length-17), and the
diagnostic ``body4d_native_image_w/h``, ``body4d_focal_first_dancer``,
``body4d_phmr_canvas_w/h``. If the file does not exist yet, we create
it; if it does (PHMR side ran first), we merge in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from threed.compare.joints import MHR70_TO_COCO17
from threed.sidecar_promthmr.reproject_vs_vitpose import load_vitpose_padded


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested individually)
# ---------------------------------------------------------------------------


def read_native_frame_size(frames_dir: Path) -> Tuple[int, int]:
    """Return ``(W, H)`` of the first JPG in ``frames_dir``.

    Raises ``FileNotFoundError`` when the directory has no JPGs — this
    is the right behaviour for the orchestrator (we don't want to
    silently fall back to a hard-coded resolution).
    """
    from PIL import Image
    candidates = sorted(frames_dir.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"no JPG frames in {frames_dir}")
    with Image.open(candidates[0]) as img:
        return int(img.width), int(img.height)


def load_body4d_focal_cam_t_per_frame(
    focal_dir: Path,
    *,
    pid: int,
    n_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack per-frame ``focal_length`` + ``camera`` from the JSON tree.

    Returns ``(focals, cam_ts)`` where ``focals`` is ``(n_frames,)`` and
    ``cam_ts`` is ``(n_frames, 3)``. Missing files / missing pid dir
    yield NaN — we don't want a quietly missing frame to look like
    "perfectly projected at the origin".
    """
    focals = np.full((n_frames,), np.nan, dtype=np.float64)
    cam_ts = np.full((n_frames, 3), np.nan, dtype=np.float64)
    pid_dir = focal_dir / str(pid)
    if not pid_dir.is_dir():
        return focals, cam_ts
    for t in range(n_frames):
        fp = pid_dir / f"{t:08d}.json"
        if not fp.is_file():
            continue
        try:
            d = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        focals[t] = float(d.get("focal_length", np.nan))
        cam = d.get("camera", None)
        if cam is not None and len(cam) == 3:
            cam_ts[t] = np.asarray(cam, dtype=np.float64)
    return focals, cam_ts


def body4d_joints_to_image_2d(
    joints_local: np.ndarray,
    *,
    focals: np.ndarray,
    cam_ts: np.ndarray,
    cx: float,
    cy: float,
    joint_index_subset: Sequence[int],
    min_depth: float = 1e-6,
) -> np.ndarray:
    """Project Body4D's per-dancer canonical joints to 2-D pixels.

    ``joints_local``: ``(T, N, J, 3)`` — usually ``J=70`` MHR.
    ``focals``: ``(T, N)`` per-(frame, dancer) focal in native pixels.
    ``cam_ts``: ``(T, N, 3)`` per-(frame, dancer) translation; the
    cam-frame coordinate of joint ``j`` is ``joints_local[..., j, :] +
    cam_ts``.
    ``joint_index_subset``: list of indices into ``J`` selecting which
    output joints to project (e.g. :data:`MHR70_TO_COCO17` for 17).

    Returns ``(T, N, len(subset), 2)``. Any joint with non-finite
    inputs (NaN focal/cam_t/joint coord) or with cam-frame ``Z <
    min_depth`` becomes NaN.
    """
    joints_local = np.asarray(joints_local, dtype=np.float64)
    focals = np.asarray(focals, dtype=np.float64)
    cam_ts = np.asarray(cam_ts, dtype=np.float64)
    T, N = focals.shape
    K = len(joint_index_subset)
    sub = list(joint_index_subset)

    # (T, N, K, 3) cam-frame joints
    selected = joints_local[..., sub, :]  # (T, N, K, 3)
    cam_t_b = cam_ts[:, :, None, :]  # (T, N, 1, 3)
    cam = selected + cam_t_b  # (T, N, K, 3)

    Z = cam[..., 2]
    X = cam[..., 0]
    Y = cam[..., 1]
    f = focals[:, :, None]  # (T, N, 1)

    valid = np.isfinite(f) & np.isfinite(cam).all(axis=-1) & (Z > min_depth)
    safe_Z = np.where(valid, Z, 1.0)
    u = np.where(valid, f * X / safe_Z + cx, np.nan)
    v = np.where(valid, f * Y / safe_Z + cy, np.nan)
    return np.stack([u, v], axis=-1)


def scale_vitpose_to_native(
    vit: np.ndarray,
    *,
    phmr_canvas_wh: Tuple[float, float],
    native_wh: Tuple[float, float],
) -> np.ndarray:
    """Scale ViTPose ``(..., 17, 3)`` from PHMR canvas to native pixels.

    The PHMR canvas is whatever PromptHMR-Vid resized the input video
    to (e.g. 504×896 for portrait). The first two channels are pixel
    coords and get rescaled; the third (confidence) is preserved.
    NaN-safe.
    """
    out = np.asarray(vit, dtype=np.float64).copy()
    sx = float(native_wh[0]) / float(phmr_canvas_wh[0])
    sy = float(native_wh[1]) / float(phmr_canvas_wh[1])
    out[..., 0] = out[..., 0] * sx
    out[..., 1] = out[..., 1] * sy
    return out


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompthmr-dir", type=Path, required=True,
        help="Stage C1 output directory (must contain results.pkl).",
    )
    p.add_argument(
        "--body4d-dir", type=Path, required=True,
        help="Stage C2 output directory (must contain joints_world.npy and "
             "focal_4d_individual/).",
    )
    p.add_argument(
        "--frames-dir", type=Path, required=True,
        help="Native-resolution frame dir (intermediates/frames_full/).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write/extend reproj_metrics.json.",
    )
    p.add_argument(
        "--vitpose-conf-threshold", type=float, default=0.3,
        help="Per-keypoint ViTPose confidence threshold; below this the 2-D "
             "ground truth is masked as NaN before computing MPJPE.",
    )
    args = p.parse_args(argv)

    phmr_dir = args.prompthmr_dir.expanduser().resolve()
    b4d_dir = args.body4d_dir.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()

    results_path = phmr_dir / "results.pkl"
    joints_path = b4d_dir / "joints_world.npy"
    focal_dir = b4d_dir / "focal_4d_individual"

    for label, path in (("results.pkl", results_path), ("joints_world.npy", joints_path),
                        ("focal_4d_individual/", focal_dir)):
        if not path.exists():
            print(f"[reproj-b4d] ERROR: missing {label}: {path}", file=sys.stderr)
            return 2

    native_w, native_h = read_native_frame_size(frames_dir)
    cx_n, cy_n = native_w / 2.0, native_h / 2.0
    print(f"[reproj-b4d] native frame size: {native_w}x{native_h}, "
          f"cx={cx_n} cy={cy_n}")

    import joblib
    results = joblib.load(results_path)
    cam = results["camera"]
    phmr_focal = float(cam["img_focal"])
    phmr_cx, phmr_cy = float(cam["img_center"][0]), float(cam["img_center"][1])
    phmr_canvas_w, phmr_canvas_h = phmr_cx * 2.0, phmr_cy * 2.0
    print(f"[reproj-b4d] PHMR canvas: {phmr_canvas_w:.0f}x{phmr_canvas_h:.0f} "
          f"(focal={phmr_focal})")

    sorted_tids = sorted(results["people"].keys())
    n_dancers = len(sorted_tids)
    joints_local = np.load(joints_path)  # (T, N_b4d, 70, 3)
    T = joints_local.shape[0]
    N_b4d = joints_local.shape[1]
    if N_b4d != n_dancers:
        print(f"[reproj-b4d] WARN: PHMR has {n_dancers} tracks but Body4D dumped "
              f"{N_b4d} dancers; using min({n_dancers}, {N_b4d})", file=sys.stderr)
    N = min(n_dancers, N_b4d)
    joints_local = joints_local[:, :N]
    sorted_tids_used = sorted_tids[:N]
    print(f"[reproj-b4d] joints_world {joints_local.shape}, "
          f"sorted tids (used): {sorted_tids_used}")

    focals = np.full((T, N), np.nan, dtype=np.float64)
    cam_ts = np.full((T, N, 3), np.nan, dtype=np.float64)
    for di in range(N):
        pid = di + 1  # body4d slot indexing convention (matches PLY tree)
        fc, ct = load_body4d_focal_cam_t_per_frame(focal_dir, pid=pid, n_frames=T)
        focals[:, di] = fc
        cam_ts[:, di] = ct
    n_missing = int(np.isnan(focals).sum())
    print(f"[reproj-b4d] focal/cam_t loaded; {n_missing} (frame, dancer) pairs missing")

    body4d_2d = body4d_joints_to_image_2d(
        joints_local,
        focals=focals,
        cam_ts=cam_ts,
        cx=cx_n,
        cy=cy_n,
        joint_index_subset=MHR70_TO_COCO17,
    )
    print(f"[reproj-b4d] body4d 2D {body4d_2d.shape} ranges: "
          f"X[{np.nanmin(body4d_2d[..., 0]):.1f}, {np.nanmax(body4d_2d[..., 0]):.1f}], "
          f"Y[{np.nanmin(body4d_2d[..., 1]):.1f}, {np.nanmax(body4d_2d[..., 1]):.1f}]")

    vit = load_vitpose_padded(results["people"], n_frames=T,
                              n_dancers=N, sorted_tids=sorted_tids_used)
    vit_native = scale_vitpose_to_native(
        vit,
        phmr_canvas_wh=(phmr_canvas_w, phmr_canvas_h),
        native_wh=(native_w, native_h),
    )
    conf = vit_native[..., 2]
    vit_uv = vit_native[..., :2].copy()
    low_conf_mask = conf < args.vitpose_conf_threshold
    n_low = int(low_conf_mask.sum())
    vit_uv[low_conf_mask] = np.nan
    print(f"[reproj-b4d] masked {n_low} low-confidence ViTPose keypoints "
          f"(threshold={args.vitpose_conf_threshold})")

    diff = np.linalg.norm(body4d_2d - vit_uv, axis=-1)  # (T, N, 17)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_joint = np.nanmean(diff, axis=(0, 1))  # (17,)
        per_dancer = np.nanmean(diff, axis=(0, 2))  # (N,)
        mean = float(np.nanmean(diff))
    print(f"[reproj-b4d] mean BODY4D vs ViTPose pixel MPJPE = {mean:.3f} px")

    # Read existing metrics if present (PHMR side may have run first), merge
    metrics: dict = {}
    if args.output.is_file():
        try:
            metrics = json.loads(args.output.read_text())
        except (OSError, json.JSONDecodeError):
            metrics = {}
    metrics.setdefault("schema_version", 1)
    metrics["body4d_native_image_w"] = int(native_w)
    metrics["body4d_native_image_h"] = int(native_h)
    metrics["body4d_phmr_canvas_w"] = float(phmr_canvas_w)
    metrics["body4d_phmr_canvas_h"] = float(phmr_canvas_h)
    metrics["body4d_focal_first_dancer"] = float(np.nanmedian(focals[:, 0])) if N >= 1 else None
    metrics["body4d_n_missing_focal_jsons"] = n_missing
    metrics["body4d_n_low_confidence_keypoints"] = n_low
    metrics["per_joint_mpjpe_body4d_vs_vitpose_px"] = per_joint.tolist()
    metrics["per_dancer_mpjpe_body4d_vs_vitpose_px"] = per_dancer.tolist()
    metrics["mean_mpjpe_body4d_vs_vitpose_px"] = mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, default=_json_default))
    print(f"[reproj-b4d] wrote {args.output}")
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
