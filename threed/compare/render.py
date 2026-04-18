"""Side-by-side overlay renderer for the dual 3D pipeline (plan Task 11c).

Stitches PromptHMR-Vid's overlay frames + SAM-Body4D's overlay frames
into a single MP4 with a 10-px gutter between them. Each pipeline has
already rendered its own per-frame overlay during Stage C:

- **PromptHMR-Vid**: writes ``hps_video.mp4`` and a per-frame overlay
  directory (depending on cfg). For Stage D we point at the
  per-frame folder if available, otherwise leave the left panel blank.
- **SAM-Body4D**: writes ``rendered_frames/<frame:08d>.jpg`` with all
  meshes overlaid (the runner used this for the Task 10 smoke test).

Frames whose left counterpart is missing (PromptHMR didn't render that
frame index) are left blank so the MP4 length always matches the
SAM-Body4D frame count. Frame ordering is the lexicographic sort of
the body4d filenames (which are zero-padded by Stage C2).

Pure helpers (:func:`stitch_side_by_side`, :func:`resize_keep_ratio`)
are unit-tested locally; the main IO loop is exercised by the Stage D
smoke test on the Lambda box.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def stitch_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    gutter_px: int = 10,
) -> np.ndarray:
    """Place two equal-sized BGR images horizontally with a black gutter.

    Both inputs must have identical shape ``(H, W, 3)``. The output is
    ``(H, W*2 + gutter_px, 3)``. Gutter pixels are zero (black).

    Used by the main loop after both source images have been resized
    to a common (H, W) via :func:`resize_keep_ratio`. Kept separate so
    we can unit-test the canvas math without involving cv2 IO.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"stitch_side_by_side requires identical size, got {left.shape} vs {right.shape}"
        )
    H, W = left.shape[:2]
    out_W = W * 2 + gutter_px
    canvas = np.zeros((H, out_W, 3), dtype=left.dtype)
    canvas[:, :W] = left
    if gutter_px > 0:
        canvas[:, W + gutter_px:] = right
    else:
        canvas[:, W:] = right
    return canvas


def resize_keep_ratio(
    img: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize ``img`` into a ``(target_h, target_w)`` BGR canvas, letterboxing.

    Maintains aspect ratio: scales the image so the larger of the two
    dimensions matches the target, then centres on a black canvas.
    Necessary because PromptHMR's overlay video may be at a different
    resolution than SAM-Body4D's per-frame JPGs (PromptHMR caps at
    896 px on the long side; Body4D uses original resolution).

    Returns a uint8 BGR array of exact shape ``(target_h, target_w, 3)``.
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_y = (target_h - new_h) // 2
    off_x = (target_w - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def _write_video(
    body4d_paths: Sequence[Path],
    prompthmr_dir: Optional[Path],
    out_path: Path,
    fps: int,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> int:
    """Internal IO loop: stitch & write the side-by-side MP4. Returns 0 on success."""
    sample = cv2.imread(str(body4d_paths[0]))
    if sample is None:
        raise RuntimeError(f"could not read sample frame {body4d_paths[0]}")
    H = target_h or sample.shape[0]
    W = target_w or sample.shape[1]
    out_W = W * 2 + 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, H))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {out_path}")

    for body_path in body4d_paths:
        body = cv2.imread(str(body_path))
        if body is None:
            print(f"[render] WARN: skipping unreadable {body_path}")
            continue
        body = resize_keep_ratio(body, H, W)

        if prompthmr_dir is not None:
            phmr_path = prompthmr_dir / body_path.name
            if phmr_path.is_file():
                phmr = cv2.imread(str(phmr_path))
                phmr = resize_keep_ratio(phmr, H, W)
            else:
                phmr = np.zeros_like(body)
        else:
            phmr = np.zeros_like(body)

        canvas = stitch_side_by_side(phmr, body, gutter_px=10)
        cv2.putText(
            canvas, "PromptHMR-Vid", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            canvas, "SAM-Body4D", (W + 30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        vw.write(canvas)

    vw.release()
    print(f"[render] wrote {out_path} ({len(body4d_paths)} frames, {out_W}x{H} @ {fps} fps)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompthmr-frames-dir", type=Path, default=None,
        help="Optional dir of PromptHMR per-frame overlays. If omitted, the "
             "left panel is left blank (useful for partial runs).",
    )
    p.add_argument(
        "--body4d-frames-dir", type=Path, required=True,
        help="Dir of SAM-Body4D per-frame overlays "
             "(<sam_body4d_dir>/rendered_frames/).",
    )
    p.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--target-h", type=int, default=None,
        help="Override panel height (defaults to body4d frame H).",
    )
    p.add_argument(
        "--target-w", type=int, default=None,
        help="Override panel width (defaults to body4d frame W).",
    )
    args = p.parse_args(argv)

    body4d_paths = sorted(args.body4d_frames_dir.glob("*.jpg"))
    if not body4d_paths:
        print(f"[render] ERROR: no JPG frames under {args.body4d_frames_dir}")
        return 2
    return _write_video(
        body4d_paths=body4d_paths,
        prompthmr_dir=args.prompthmr_frames_dir,
        out_path=args.output,
        fps=args.fps,
        target_h=args.target_h,
        target_w=args.target_w,
    )


if __name__ == "__main__":
    raise SystemExit(main())
