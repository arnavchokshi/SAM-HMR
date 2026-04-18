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


def resize_palette_canvas(canvas: np.ndarray, *, dst_h: int, dst_w: int) -> np.ndarray:
    """Nearest-neighbour resize of a palette canvas (pixel == tid).

    Used so SAM-Body4D's mask reshape lines up with frames_full when
    PHMR's input frames were downscaled to ``max_height=896``. NN
    interpolation preserves tid values exactly (no blending of adjacent
    tids); slight aliasing on dancer edges is harmless because
    SAM-Body4D rebuilds person masks from the palette anyway.

    No-op when ``(dst_h, dst_w)`` already matches the input shape.
    """
    h, w = canvas.shape[:2]
    if (h, w) == (dst_h, dst_w):
        return canvas
    import cv2
    out = cv2.resize(canvas, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8, copy=False)


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


def chdir_to_prompthmr(prompthmr_path: Path) -> Path:
    """Chdir to the PromptHMR clone root and return the previous cwd.

    Necessary because PromptHMR's ``pipeline/__init__.py`` transitively imports
    ``pipeline.phmr_vid`` which does ``sys.path.insert(0, 'pipeline/gvhmr')``
    — a *relative* path that only resolves to the right place when cwd is the
    PromptHMR root. Hydra's SAM-2 config lookup
    (``pipeline/sam2/sam2_hiera_t.yaml``) is also relative.

    Returns the previous cwd so callers can restore it if needed.
    """
    previous = Path.cwd()
    os.chdir(prompthmr_path)
    return previous


def hydra_absolute_config_name(prompthmr_path: Path, cfg_relpath: str) -> str:
    """Convert a PromptHMR-relative config path into Hydra's '//abs/path' form.

    Hydra's ``compose(config_name=…)`` looks up names in its search path
    (``pkg://sam2`` by default — the installed sam2 package, NOT PromptHMR's
    local ``pipeline/sam2/`` directory). PromptHMR's local
    ``pipeline/sam2/sam2_hiera_t.yaml`` differs from the upstream sam2
    package's ``sam2_hiera_t.yaml`` (e.g. feat_sizes [32,32] vs [64,64])
    and the local config is required for PromptHMR's custom
    SAM2VideoPredictor subclass to load cleanly.

    The trick (lifted verbatim from ``pipeline/tools.py``) is to pass
    ``'/' + os.path.abspath(cfg_relpath)`` — the leading double slash
    makes Hydra treat the rest as a filesystem-absolute path instead of
    a search-path lookup.
    """
    abs_cfg = os.path.abspath(prompthmr_path / cfg_relpath)
    return "/" + abs_cfg


def _build_predictor(prompthmr_path: Path, ckpt: Path, cfg: str):
    """Construct the SAM-2 video predictor (GPU).

    Side-effects: sys.path is mutated; cwd is changed to ``prompthmr_path``.
    Caller is responsible for resolving every other path argument to an
    absolute path before calling this. ``cfg`` is the PromptHMR-relative
    config path (e.g. ``pipeline/sam2/sam2_hiera_t.yaml``); it is converted
    to Hydra's absolute-path form internally.
    """
    inject_prompthmr_path(prompthmr_path)
    chdir_to_prompthmr(prompthmr_path)
    abs_cfg = hydra_absolute_config_name(prompthmr_path, cfg)
    from pipeline.detector.sam2_video_predictor import (
        build_sam2_video_predictor,
    )
    return build_sam2_video_predictor(abs_cfg, str(ckpt))


def load_video_frames_bgr(frames_dir: Path) -> np.ndarray:
    """Load every ``*.jpg`` in ``frames_dir`` (sorted) into ``(N, H, W, 3)`` BGR uint8.

    PromptHMR's ``SAM2VideoPredictor.init_state`` was modified upstream to
    require ``video_frames`` as the primary positional argument, replacing
    the original SAM-2 ``video_path`` API. ``video_frames`` must be a
    stacked numpy array (``init_state`` does ``video_frames.shape[1:3]``)
    and the in-class ``_load_img_as_tensor`` accepts BGR numpy arrays
    without channel-swap, so we load via ``cv2.imread`` and stack.
    """
    import cv2

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no JPG frames in {frames_dir}")
    return np.stack([cv2.imread(str(p)) for p in frame_paths])


def _propagate_with_predictor(
    predictor,
    frames_dir: Path,
    tracks: Dict,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Run SAM-2 propagation, returning ``{frame -> {tid -> bool mask}}``."""
    import torch

    video_frames = load_video_frames_bgr(frames_dir)
    state = predictor.init_state(
        video_frames=video_frames,
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
    *,
    output_size: Optional[Tuple[int, int]] = None,
) -> None:
    """Write a palette PNG per frame (pixel value == tid) for SAM-Body4D
    consumers.

    ``output_size`` (default ``None``) optionally upscales each canvas to
    ``(H_out, W_out)`` via nearest-neighbour so SAM-Body4D's mask
    reshape against frames_full lines up. Required when PHMR's input
    frames were downscaled to ``max_height=896`` but body4d reads the
    full-resolution images.
    """
    from PIL import Image

    out_palette_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(n_frames):
        tid_masks = {
            tid: msk
            for tid, msk in per_frame_per_tid.get(frame, {}).items()
            if tid in valid_frames and frame in valid_frames[tid]
        }
        canvas = assemble_palette_canvas(tid_masks, H, W)
        if output_size is not None:
            out_h, out_w = output_size
            canvas = resize_palette_canvas(canvas, dst_h=out_h, dst_w=out_w)
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

    # Resolve every path to absolute BEFORE _build_predictor() does its chdir,
    # otherwise relative paths (especially intermediates-dir from a non-PromptHMR
    # cwd) silently get re-rooted under PromptHMR.
    interm = args.intermediates_dir.expanduser().resolve()
    prompthmr_path = args.prompthmr_path.expanduser().resolve()
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
        sam2_ckpt, _ = resolve_default_sam2_paths(prompthmr_path)
    else:
        sam2_ckpt = args.sam2_checkpoint.expanduser().resolve()
    sam2_cfg = args.sam2_config
    if not sam2_ckpt.is_file():
        raise SystemExit(f"sam2 checkpoint not found: {sam2_ckpt}")

    import cv2

    tracks = joblib.load(tracks_pkl)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_frames = len(frame_paths)
    if n_frames == 0:
        raise SystemExit(f"no JPG frames in {frames_dir}")
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    print(
        f"[build_masks] {n_frames} frames @ {W}x{H}, {len(tracks)} tids, "
        f"sam2 ckpt={sam2_ckpt.name}, cfg={sam2_cfg}"
    )

    predictor = _build_predictor(prompthmr_path, sam2_ckpt, sam2_cfg)
    per_frame_per_tid = _propagate_with_predictor(predictor, frames_dir, tracks)

    full_frames_dir = interm / "frames_full"
    full_size: Optional[Tuple[int, int]] = None
    full_paths = sorted(full_frames_dir.glob("*.jpg"))
    if full_paths:
        H_full, W_full = cv2.imread(str(full_paths[0])).shape[:2]
        if (H_full, W_full) != (H, W):
            full_size = (H_full, W_full)
            print(
                f"[build_masks] frames_full at {W_full}x{H_full} differs from "
                f"frames at {W}x{H}; palette PNGs will be NN-upscaled for body4d"
            )

    vf = valid_frames_set(tracks)
    n_per_tid = _write_per_track_pngs(out_per_tid, per_frame_per_tid, vf, n_frames)
    _write_palette_pngs(
        out_palette, per_frame_per_tid, vf, n_frames, H, W,
        output_size=full_size,
    )
    union = compute_union(per_frame_per_tid, n_frames, H, W)
    np.save(out_union, union)
    print(
        f"[build_masks] wrote {n_per_tid} per-tid PNGs, {n_frames} palette PNGs"
        + (f" (palette NN-upscaled to {full_size[1]}x{full_size[0]})" if full_size else "")
        + f", union shape={tuple(union.shape)} sum={int(union.sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
