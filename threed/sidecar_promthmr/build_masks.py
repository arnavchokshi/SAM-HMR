"""Per-track SAM-2 mask propagation seeded by our DeepOcSort bboxes.

Plan Task 6 — runs in the ``phmr_pt2.4`` conda env on the box, importing
PromptHMR's bundled SAM-2 video predictor via ``$PROMPTHMR_PATH``.

Reads (under ``--intermediates-dir``):
    frames/             # JPGs at max_height = CompareConfig.max_height (896)
    tracks.pkl          # threed.io.save_tracks payload
                        #   {tid -> {track_id, frames, bboxes,
                        #             confs, masks?, detected?}}

Writes (under same dir):
    masks_per_track/{tid}/{frame:08d}.png   # binary 0/255 PNG, only frames
                                            # where DeepOcSort had this tid
    masks_palette/{frame:08d}.png           # palette PNG, pixel value == tid
                                            # (overlap resolved by larger-tid-wins);
                                            # SAM-Body4D consumes this format
    masks_union.npy                         # (T, H, W) bool union of all tids
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np


def davis_palette() -> bytes:
    """DAVIS-style HSV-cycled 256-entry palette (3 bytes per entry).

    Index 0 is reserved as background (black). Indices 1..255 cycle through
    HSV with saturation/value 0.9 to give visually-distinct colours that
    SAM-Body4D's palette renderer recognises.
    """
    palette = [0, 0, 0]
    for i in range(1, 256):
        h = i * 360 / 256
        s = 0.9
        v = 0.9
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0.0
        elif h < 120:
            r, g, b = x, c, 0.0
        elif h < 180:
            r, g, b = 0.0, c, x
        elif h < 240:
            r, g, b = 0.0, x, c
        elif h < 300:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        palette.extend([
            int(round((r + m) * 255)),
            int(round((g + m) * 255)),
            int(round((b + m) * 255)),
        ])
    return bytes(palette)


DAVIS_PALETTE = davis_palette()


def resolve_default_sam2_paths(prompthmr_path: Path) -> Tuple[Path, str]:
    """Return ``(default sam2 ckpt, default config string)`` inside PromptHMR.

    Default to ``sam2_hiera_tiny.pt`` because that is the only SAM-2 weight
    that ``scripts/fetch_data.sh`` actually downloads. Override via CLI
    ``--sam2-checkpoint`` / ``--sam2-config`` if you have larger weights.
    """
    return (
        prompthmr_path / "data" / "pretrain" / "sam2_ckpts" / "sam2_hiera_tiny.pt",
        "pipeline/sam2/sam2_hiera_t.yaml",
    )


def valid_frames_set(tracks: Dict) -> Dict[int, set]:
    """Return ``{tid -> set of frame indices where DeepOcSort had this tid}``."""
    out: Dict[int, set] = {}
    for tid, t in tracks.items():
        frames = t["frames"]
        out[int(tid)] = {int(f) for f in (frames.tolist() if hasattr(frames, "tolist") else frames)}
    return out


def assemble_palette_canvas(
    tid_to_mask: Dict[int, np.ndarray],
    H: int,
    W: int,
) -> np.ndarray:
    """Combine per-tid bool masks into a uint8 canvas where pixel value == tid.

    Overlap resolved deterministically by larger-tid-wins (we iterate ascending
    so larger tids overwrite smaller ones). Empty input gives an all-zero
    background canvas.
    """
    canvas = np.zeros((H, W), dtype=np.uint8)
    for tid in sorted(tid_to_mask.keys()):
        canvas[tid_to_mask[tid]] = int(tid)
    return canvas


def compute_union(
    per_frame_per_tid: Dict[int, Dict[int, np.ndarray]],
    n_frames: int,
    H: int,
    W: int,
) -> np.ndarray:
    """OR-combine per-tid masks across all tids to produce ``(T, H, W)`` bool."""
    union = np.zeros((n_frames, H, W), dtype=bool)
    for frame, tid_masks in per_frame_per_tid.items():
        if not (0 <= frame < n_frames):
            continue
        for msk in tid_masks.values():
            union[frame] |= msk
    return union


def inject_prompthmr_path(prompthmr_path: Path) -> None:
    """Insert ``<prompthmr_path>`` at the front of ``sys.path`` (idempotent).

    Required so ``from pipeline.detector.sam2_video_predictor import …``
    resolves to the PromptHMR clone.
    """
    p = str(Path(prompthmr_path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_predictor(prompthmr_path: Path, ckpt: Path, cfg: str):
    """Construct the SAM-2 video predictor (GPU). Side-effect: sys.path mutated."""
    inject_prompthmr_path(prompthmr_path)
    from pipeline.detector.sam2_video_predictor import (
        build_sam2_video_predictor,
    )
    return build_sam2_video_predictor(cfg, str(ckpt))


def _propagate_with_predictor(
    predictor,
    frames_dir: Path,
    tracks: Dict,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Run SAM-2 propagation, returning ``{frame -> {tid -> bool mask}}``."""
    import torch

    state = predictor.init_state(
        video_path=str(frames_dir),
        async_loading_frames=False,
        offload_video_to_cpu=True,
    )
    for tid, t in tracks.items():
        frames = t["frames"]
        if len(frames) == 0:
            continue
        first_frame = int(frames[0])
        x1, y1, x2, y2 = (float(v) for v in t["bboxes"][0])
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        predictor.add_new_points_or_box(
            state,
            frame_idx=first_frame,
            obj_id=int(tid),
            box=box,
        )

    out: Dict[int, Dict[int, np.ndarray]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame, obj_ids, mask_logits in predictor.propagate_in_video(state):
            tid_masks: Dict[int, np.ndarray] = {}
            for i, oid in enumerate(obj_ids):
                logits = mask_logits[i]
                if logits.dim() == 3:
                    logits = logits[0]
                tid_masks[int(oid)] = (logits > 0.0).cpu().numpy().astype(bool)
            out[int(frame)] = tid_masks
    return out


def _write_per_track_pngs(
    out_per_tid_dir: Path,
    per_frame_per_tid: Dict[int, Dict[int, np.ndarray]],
    valid_frames: Dict[int, set],
    n_frames: int,
) -> int:
    """Write binary 0/255 PNGs per (tid, frame) pair where DeepOcSort had the tid."""
    import cv2

    out_per_tid_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for tid in valid_frames.keys():
        out_t = out_per_tid_dir / str(tid)
        out_t.mkdir(parents=True, exist_ok=True)
        for frame in range(n_frames):
            if frame not in valid_frames[tid]:
                continue
            tid_masks = per_frame_per_tid.get(frame, {})
            if tid not in tid_masks:
                continue
            cv2.imwrite(
                str(out_t / f"{frame:08d}.png"),
                (tid_masks[tid].astype(np.uint8) * 255),
            )
            written += 1
    return written


def _write_palette_pngs(
    out_palette_dir: Path,
    per_frame_per_tid: Dict[int, Dict[int, np.ndarray]],
    valid_frames: Dict[int, set],
    n_frames: int,
    H: int,
    W: int,
) -> None:
    """Write a palette PNG per frame (pixel value == tid) for SAM-Body4D consumers."""
    from PIL import Image

    out_palette_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(n_frames):
        tid_masks = {
            tid: msk
            for tid, msk in per_frame_per_tid.get(frame, {}).items()
            if tid in valid_frames and frame in valid_frames[tid]
        }
        canvas = assemble_palette_canvas(tid_masks, H, W)
        img = Image.fromarray(canvas, mode="P")
        img.putpalette(DAVIS_PALETTE)
        img.save(str(out_palette_dir / f"{frame:08d}.png"))


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--intermediates-dir",
        type=Path,
        required=True,
        help="Path to <clip>/intermediates/ produced by Stage A",
    )
    p.add_argument(
        "--prompthmr-path",
        type=Path,
        default=Path(os.environ.get("PROMPTHMR_PATH", "~/code/PromptHMR")).expanduser(),
        help="Where the PromptHMR clone lives (defaults to $PROMPTHMR_PATH or ~/code/PromptHMR)",
    )
    p.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=None,
        help="SAM-2 weight; defaults to <prompthmr-path>/data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt",
    )
    p.add_argument(
        "--sam2-config",
        type=str,
        default="pipeline/sam2/sam2_hiera_t.yaml",
        help="Hydra config relative to the PromptHMR repo (default: tiny config)",
    )
    args = p.parse_args(argv)

    interm = args.intermediates_dir.expanduser().resolve()
    frames_dir = interm / "frames"
    tracks_pkl = interm / "tracks.pkl"
    out_per_tid = interm / "masks_per_track"
    out_palette = interm / "masks_palette"
    out_union = interm / "masks_union.npy"

    if not frames_dir.is_dir():
        raise SystemExit(f"frames dir not found: {frames_dir}")
    if not tracks_pkl.is_file():
        raise SystemExit(f"tracks.pkl not found: {tracks_pkl}")

    if args.sam2_checkpoint is None:
        args.sam2_checkpoint, _ = resolve_default_sam2_paths(args.prompthmr_path)
    if not args.sam2_checkpoint.is_file():
        raise SystemExit(f"sam2 checkpoint not found: {args.sam2_checkpoint}")

    import cv2

    tracks = joblib.load(tracks_pkl)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_frames = len(frame_paths)
    if n_frames == 0:
        raise SystemExit(f"no JPG frames in {frames_dir}")
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    print(
        f"[build_masks] {n_frames} frames @ {W}x{H}, {len(tracks)} tids, "
        f"sam2 ckpt={args.sam2_checkpoint.name}, cfg={args.sam2_config}"
    )

    predictor = _build_predictor(args.prompthmr_path, args.sam2_checkpoint, args.sam2_config)
    per_frame_per_tid = _propagate_with_predictor(predictor, frames_dir, tracks)

    vf = valid_frames_set(tracks)
    n_per_tid = _write_per_track_pngs(out_per_tid, per_frame_per_tid, vf, n_frames)
    _write_palette_pngs(out_palette, per_frame_per_tid, vf, n_frames, H, W)
    union = compute_union(per_frame_per_tid, n_frames, H, W)
    np.save(out_union, union)
    print(
        f"[build_masks] wrote {n_per_tid} per-tid PNGs, {n_frames} palette PNGs, "
        f"union shape={tuple(union.shape)} sum={int(union.sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
