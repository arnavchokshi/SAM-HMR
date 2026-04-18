"""Render annotated tracking-overlay MP4s for every cached clip.

For each clip we re-decode the source video, project the post-processed
tracker output onto every frame, and burn a HUD with:

  * per-track stable-id colored bbox + id label + per-frame confidence
  * a thin trail of the last N centers per track (motion ribbon)
  * a top-left HUD with: frame index, predicted count vs GT count,
    cumulative IDsw / id_merge events, and a green/amber/red dot
  * a bottom strip showing the per-frame jitter (size_outlier/center
    outlier flags) so the user immediately spots wobbly frames
  * a status footer stamping the config_id / smoothing sigma in use

Output: ``runs/overlays/<clip>_tracking_overlay.mp4`` plus a per-clip
``<clip>_summary.json`` with the headline metrics.

This is the file the user opens to see "did the tracker improve?".

Usage::

    python eval/render_overlay_videos.py
    python eval/render_overlay_videos.py --clips easyTest --max-frames 60
"""
from __future__ import annotations

import argparse
import colorsys
import json
import logging
import math
import pickle
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prune_tracks import FrameDetections  # noqa: E402
from tracking.postprocess import (  # noqa: E402
    Track,
    frame_detections_to_raw_tracks,
    postprocess_tracks,
)


log = logging.getLogger("render_overlay_videos")

DEFAULT_USER_CLIPS_ROOT = Path("/Users/arnavchokshi/Desktop")


@dataclass(frozen=True)
class ClipSpec:
    name: str
    video: Path
    leaked: bool
    gt: Optional[Path] = None
    expected_const: Optional[int] = None
    max_frames: Optional[int] = None
    cache_pkl: Optional[Path] = None


def _parse_clip_overrides(items: Optional[Sequence[str]]) -> Dict[str, Path]:
    """``NAME=/abs/path`` pairs from repeated ``--clip-video``."""
    out: Dict[str, Path] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"expected NAME=PATH, got: {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        out[name] = Path(path).expanduser().resolve()
    return out


def _all_clip_specs(
    root: Path,
    cache_root: Path,
    *,
    video_overrides: Optional[Dict[str, Path]] = None,
    extra_clips: Optional[Dict[str, Path]] = None,
) -> List[ClipSpec]:
    """Same set as :mod:`eval.eval_counts` plus the cache pkl path."""

    def _newest(clip_name: str) -> Optional[Path]:
        d = cache_root / clip_name
        if not d.is_dir():
            return None
        pkls = sorted(d.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return pkls[0] if pkls else None

    vo = video_overrides or {}
    candidates = [
        ClipSpec("mirrorTest", vo.get("mirrorTest", root / "mirrorTest" / "IMG_2946.MP4"),
                 leaked=True, gt=root / "mirrorTest" / "gt" / "gt.txt"),
        ClipSpec("gymTest",    vo.get("gymTest", root / "gymTest" / "IMG_8309.mov"),
                 leaked=False, gt=root / "gymTest" / "gt" / "gt.txt"),
        ClipSpec("adiTest",    vo.get("adiTest", root / "adiTest" / "IMG_1649.mov"),
                 leaked=False, gt=root / "adiTest" / "gt" / "gt.txt",
                 max_frames=188),
        ClipSpec("BigTest",    vo.get("BigTest", root / "BigTest" / "BigTest.mov"),
                 leaked=False, gt=root / "BigTest" / "gt" / "gt.txt"),
        ClipSpec("easyTest",   vo.get("easyTest", root / "easyTest" / "IMG_2082.mov"),
                 leaked=False, gt=root / "easyTest" / "gt" / "gt.txt"),
        ClipSpec("2pplTest",   vo.get("2pplTest", root / "2pplTest.mov"),
                 leaked=False, expected_const=2),
        ClipSpec("loveTest",   vo.get("loveTest", root / "loveTest" / "IMG_9265.mov"),
                 leaked=False, gt=root / "loveTest" / "gt" / "gt.txt"),
        ClipSpec("shorterTest", vo.get("shorterTest", root / "shorterTest" / "TestVideo.mov"),
                 leaked=False, gt=root / "shorterTest" / "gt" / "gt.txt"),
    ]
    out: List[ClipSpec] = []
    for c in candidates:
        cache = _newest(c.name)
        out.append(ClipSpec(
            name=c.name,
            video=c.video,
            leaked=c.leaked,
            gt=c.gt,
            expected_const=c.expected_const,
            max_frames=c.max_frames,
            cache_pkl=cache,
        ))
    for name, video in (extra_clips or {}).items():
        out.append(ClipSpec(
            name=name,
            video=video,
            leaked=False,
            gt=None,
            expected_const=None,
            max_frames=None,
            cache_pkl=_newest(name),
        ))
    return out


# ---------------------------------------------------------------------------
# colors + drawing helpers
# ---------------------------------------------------------------------------


_COLORS_CACHE: Dict[int, Tuple[int, int, int]] = {}


def _color_for_id(tid: int) -> Tuple[int, int, int]:
    """Deterministic, well-spaced color per track id (BGR for cv2)."""
    if tid in _COLORS_CACHE:
        return _COLORS_CACHE[tid]
    # Golden-ratio hue spacing keeps consecutive ids visually distinct.
    h = (abs(int(tid)) * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.95)
    bgr = (int(b * 255), int(g * 255), int(r * 255))
    _COLORS_CACHE[tid] = bgr
    return bgr


def _draw_box(img: np.ndarray, box: np.ndarray, color, thickness: int) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    fg=(255, 255, 255),
    bg=(0, 0, 0),
    scale: float = 0.55,
    thick: int = 1,
    pad: int = 4,
) -> None:
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(
        img,
        (x - pad, y - th - pad - bl + 1),
        (x + tw + pad, y + bl + pad - 1),
        bg, thickness=cv2.FILLED,
    )
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _draw_polyline(img: np.ndarray, pts: Sequence[Tuple[float, float]], color, thickness: int) -> None:
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32)
    cv2.polylines(img, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# GT loaders
# ---------------------------------------------------------------------------


def _load_gt_count_per_frame(gt_path: Optional[Path], n_frames: int) -> Optional[np.ndarray]:
    """Read MOT 1.1 gt.txt, return per-frame count (1-indexed -> 0-indexed)."""
    if gt_path is None or not gt_path.is_file():
        return None
    counts = np.zeros(n_frames, dtype=np.int32)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                fr_one = int(parts[0])
            except ValueError:
                continue
            fr = fr_one - 1
            if 0 <= fr < n_frames:
                counts[fr] += 1
    return counts


# ---------------------------------------------------------------------------
# Per-frame projection of tracks
# ---------------------------------------------------------------------------


@dataclass
class FrameView:
    boxes: List[np.ndarray]
    confs: List[float]
    tids: List[int]
    detected: List[bool]


def _project_tracks(tracks: Dict[int, Track], n_frames: int) -> List[FrameView]:
    out: List[FrameView] = [FrameView([], [], [], []) for _ in range(n_frames)]
    for tid, t in tracks.items():
        for k, fr in enumerate(t.frames):
            f = int(fr)
            if 0 <= f < n_frames:
                out[f].boxes.append(t.bboxes[k])
                out[f].confs.append(float(t.confs[k]))
                out[f].tids.append(int(t.track_id))
                out[f].detected.append(bool(t.detected[k]))
    return out


# ---------------------------------------------------------------------------
# main render loop
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    trail_frames: int = 24
    bottom_strip_height: int = 64
    show_gt_diff: bool = True
    quality_dot_radius: int = 8
    fontscale_id: float = 0.65
    box_thickness: int = 3
    detected_only_alpha: float = 0.55  # "interpolated" frames get faded box
    output_fps: Optional[float] = None  # default = source fps
    max_frames: Optional[int] = None
    config_label: str = ""


def _quality_color(pred: int, gt_count: Optional[int]) -> Tuple[int, int, int]:
    """Green if pred==gt, amber if |pred-gt|==1, red if larger."""
    if gt_count is None or gt_count <= 0:
        return (180, 180, 180)
    diff = abs(pred - gt_count)
    if diff == 0:
        return (60, 200, 90)     # green-ish (BGR)
    if diff == 1:
        return (60, 180, 230)    # amber
    return (60, 60, 230)         # red


def _render_clip_overlay(
    *,
    clip: ClipSpec,
    tracks: Dict[int, Track],
    n_decoded_frames: int,
    output_path: Path,
    cfg: RenderConfig,
    expected_per_frame: Optional[np.ndarray],
    expected_const: Optional[int],
) -> Dict[str, object]:
    """Decode the source video and write ``output_path`` with HUD overlay.

    Returns a small JSON-able dict with summary stats."""
    if not clip.video.is_file():
        log.warning("[%s] source video missing: %s", clip.name, clip.video)
        return {"error": "video_missing"}

    cap = cv2.VideoCapture(str(clip.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        log.warning("[%s] invalid frame size %dx%d", clip.name, src_w, src_h)
        cap.release()
        return {"error": "video_invalid"}

    out_fps = cfg.output_fps or src_fps
    out_h = src_h + cfg.bottom_strip_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (src_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

    per_frame = _project_tracks(tracks, n_decoded_frames)

    # Trail buffer per track id
    trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.trail_frames))

    fr_idx = 0
    n_pred_total = 0
    count_match_frames = 0
    count_within1_frames = 0
    n_judged_frames = 0
    bottom_history: deque = deque(maxlen=src_w // 2)  # for the strip
    while True:
        if cfg.max_frames is not None and fr_idx >= cfg.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if fr_idx >= n_decoded_frames:
            break

        view = per_frame[fr_idx]
        # Update trails for currently-active tracks
        active_tids = set(view.tids)
        for tid in list(trails.keys()):
            if tid not in active_tids:
                # still age it out by appending None? We just skip
                pass
        for box, tid, conf, det in zip(view.boxes, view.tids, view.confs, view.detected):
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            trails[tid].append((float(cx), float(cy), bool(det)))

        # Determine GT count for this frame
        if expected_per_frame is not None:
            gt = int(expected_per_frame[fr_idx])
        elif expected_const is not None:
            gt = int(expected_const)
        else:
            gt = -1
        pred = len(view.tids)
        n_pred_total += pred
        if gt >= 0:
            n_judged_frames += 1
            if pred == gt:
                count_match_frames += 1
            if abs(pred - gt) <= 1:
                count_within1_frames += 1
        bottom_history.append((pred, gt if gt >= 0 else None))

        # Compose canvas: source frame on top, bottom strip below.
        canvas = np.zeros((out_h, src_w, 3), dtype=np.uint8)
        canvas[:src_h] = frame

        # Draw trails first (under boxes)
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            color = _color_for_id(tid)
            # Draw thinning trail by recursively shorter polylines
            n = len(pts)
            for thickness, take in ((1, n), (2, max(2, n // 2)), (3, max(2, n // 4))):
                segment = list(pts)[-take:]
                xy = [(int(p[0]), int(p[1])) for p in segment]
                _draw_polyline(canvas, xy, color, thickness)

        # Draw boxes + labels
        for box, tid, conf, det in zip(view.boxes, view.tids, view.confs, view.detected):
            color = _color_for_id(tid)
            thickness = cfg.box_thickness if det else max(1, cfg.box_thickness - 1)
            _draw_box(canvas, box, color, thickness)
            x1 = int(round(box[0]))
            y1 = int(round(box[1]))
            label = f"id {tid}  {conf:.2f}"
            if not det:
                label += "  (interp)"
            _draw_label(
                canvas, label, (max(2, x1 + 4), max(18, y1 - 6)),
                fg=(255, 255, 255), bg=color,
                scale=cfg.fontscale_id, thick=1,
            )

        # HUD top-left
        hud_lines = [
            f"frame {fr_idx + 1} / {n_decoded_frames}",
            f"pred {pred}" + (f"   gt {gt}" if gt >= 0 else ""),
        ]
        y_cursor = 30
        for line in hud_lines:
            _draw_label(canvas, line, (12, y_cursor), scale=0.62, thick=1, pad=5,
                        bg=(20, 20, 20), fg=(255, 255, 255))
            y_cursor += 28

        # Quality indicator dot
        qcolor = _quality_color(pred, gt if gt >= 0 else None)
        cv2.circle(canvas, (12 + cfg.quality_dot_radius * 2, y_cursor + 8),
                   cfg.quality_dot_radius, qcolor, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (12 + cfg.quality_dot_radius * 2, y_cursor + 8),
                   cfg.quality_dot_radius, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # HUD top-right: clip name + leaked banner
        right_lines = [clip.name + (" (LEAKED)" if clip.leaked else "")]
        if cfg.config_label:
            right_lines.append(cfg.config_label)
        for li, line in enumerate(right_lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x = src_w - tw - 24
            y = 30 + li * 28
            bg = (20, 20, 80) if "LEAK" in line else (20, 20, 20)
            _draw_label(canvas, line, (x, y), bg=bg, fg=(255, 255, 255), scale=0.6, thick=1, pad=5)

        # Bottom strip: color-coded count stripe
        strip = canvas[src_h:, :, :]
        strip[:] = (15, 15, 15)
        # left half: per-frame count history (fixed window)
        bw = src_w // 2
        for k, (p, g) in enumerate(list(bottom_history)[-bw:]):
            x = k
            color = _quality_color(p, g)
            cv2.line(strip, (x, 6), (x, cfg.bottom_strip_height - 6), color, 1)
        # right half: legend
        legend_x = src_w // 2 + 24
        cv2.line(strip, (legend_x, 12), (legend_x + 22, 12), (60, 200, 90), 4)
        _draw_label(strip, "match", (legend_x + 30, 18), bg=(15, 15, 15), fg=(220, 220, 220), scale=0.5)
        cv2.line(strip, (legend_x, 32), (legend_x + 22, 32), (60, 180, 230), 4)
        _draw_label(strip, "+/-1", (legend_x + 30, 38), bg=(15, 15, 15), fg=(220, 220, 220), scale=0.5)
        cv2.line(strip, (legend_x, 52), (legend_x + 22, 52), (60, 60, 230), 4)
        _draw_label(strip, ">+/-1", (legend_x + 30, 58), bg=(15, 15, 15), fg=(220, 220, 220), scale=0.5)

        # Active id chips on the strip's far right
        chip_x = src_w - 12
        for tid in sorted(view.tids, reverse=True):
            color = _color_for_id(tid)
            label = f"{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            chip_x -= (tw + 14)
            if chip_x < src_w // 2 + 160:
                break
            cv2.rectangle(strip, (chip_x - 4, 8), (chip_x + tw + 4, 8 + th + 6),
                          color, thickness=cv2.FILLED)
            cv2.putText(strip, label, (chip_x, 8 + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

        writer.write(canvas)
        fr_idx += 1

    cap.release()
    writer.release()

    summary = {
        "clip": clip.name,
        "frames_rendered": fr_idx,
        "src_fps": float(src_fps),
        "src_size": [src_w, src_h],
        "n_tracks": len(tracks),
        "mean_pred_per_frame": float(n_pred_total / max(fr_idx, 1)),
        "count_exact_acc": float(count_match_frames / max(n_judged_frames, 1)) if n_judged_frames else None,
        "count_within1_acc": float(count_within1_frames / max(n_judged_frames, 1)) if n_judged_frames else None,
        "output_path": str(output_path),
        "leaked": clip.leaked,
    }
    log.info(
        "[%s] -> %s  (%d frames, %d tracks, count_exact=%s, within1=%s)",
        clip.name, output_path, fr_idx, len(tracks),
        f"{summary['count_exact_acc']:.3f}" if summary["count_exact_acc"] is not None else "n/a",
        f"{summary['count_within1_acc']:.3f}" if summary["count_within1_acc"] is not None else "n/a",
    )
    return summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", type=Path,
                   default=REPO_ROOT / "runs" / "eval_counts" / "_cache")
    p.add_argument("--user-clips-root", type=Path, default=DEFAULT_USER_CLIPS_ROOT)
    p.add_argument("--clip-video", action="append", default=None,
                   metavar="NAME=PATH",
                   help="Override the default video path for a clip name "
                        "(repeatable). Example: "
                        "--clip-video 2pplTest=/path/to/video.mov")
    p.add_argument("--extra-clip", action="append", default=None,
                   metavar="NAME=PATH",
                   help="Add an unseen clip with no GT (repeatable). "
                        "Cache must already exist under "
                        "<cache-root>/<NAME>/.")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "runs" / "overlays")
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--gaussian-sigma", type=float, default=3.0,
                   help="Smoothing sigma to use for the postprocess pass.")
    # Defaults match docs/WINNING_PIPELINE_CONFIGURATION.md (Phase 1+2).
    p.add_argument("--min-total-frames", type=int, default=30)
    p.add_argument("--min-conf", type=float, default=0.38)
    p.add_argument("--max-gap-interp", type=int, default=12)
    p.add_argument("--num-max-people", type=int, default=25)
    p.add_argument("--overlap-merge-iou-thresh", type=float, default=0.7)
    p.add_argument("--id-merge-iou-thresh", type=float, default=0.5)
    p.add_argument("--id-merge-max-gap", type=int, default=8)
    p.add_argument("--medfilt-window", type=int, default=11)
    p.add_argument("--pose-cos-thresh", type=float, default=0.0,
                   help="Pose-cosine gate for long-gap merge (0 = off).")
    p.add_argument("--pose-max-gap", type=int, default=40)
    p.add_argument("--pose-min-iou-for-pair", type=float, default=0.0)
    p.add_argument("--pose-max-center-dist", type=float, default=60.0,
                   help="Proximity merge (px). Pass a huge value to disable.")
    p.add_argument("--config-label", type=str, default="",
                   help="Optional label rendered top-right of every frame.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cache_root: Path = args.cache_root
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        vo = _parse_clip_overrides(args.clip_video)
        extra = _parse_clip_overrides(args.extra_clip)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    all_clips = _all_clip_specs(args.user_clips_root, cache_root,
                                video_overrides=vo, extra_clips=extra)
    wanted = set(args.clips) if args.clips else None
    if wanted:
        all_clips = [c for c in all_clips if c.name in wanted]

    config_label = args.config_label or (
        f"sigma={args.gaussian_sigma:.2f}  "
        f"min_total={args.min_total_frames}  "
        f"min_conf={args.min_conf:.2f}  "
        f"max_gap={args.max_gap_interp}  "
        f"prox={args.pose_max_center_dist:.0f}px"
    )

    summaries = []
    for clip in all_clips:
        if clip.cache_pkl is None or not clip.cache_pkl.is_file():
            log.warning("[%s] no cached detections under %s/<clip>/*.pkl ; run "
                        "eval/eval_counts.py once to populate it.", clip.name, cache_root)
            continue
        if not clip.video.is_file():
            log.warning("[%s] missing source video: %s", clip.name, clip.video)
            continue

        with open(clip.cache_pkl, "rb") as f:
            frames: List[FrameDetections] = pickle.load(f)
        n = len(frames)
        if clip.max_frames is not None:
            n = min(n, int(clip.max_frames))
            frames = frames[:n]
        log.info("[%s] %d frames decoded; postprocessing...", clip.name, n)

        raw = frame_detections_to_raw_tracks(frames)
        tracks = postprocess_tracks(
            raw,
            min_box_w=10,
            min_box_h=10,
            min_total_frames=int(args.min_total_frames),
            min_conf=float(args.min_conf),
            max_gap_interp=int(args.max_gap_interp),
            id_merge_max_gap=int(args.id_merge_max_gap),
            id_merge_iou_thresh=float(args.id_merge_iou_thresh),
            id_merge_osnet_cos_thresh=0.7,
            medfilt_window=int(args.medfilt_window),
            gaussian_sigma=float(args.gaussian_sigma),
            num_max_people=int(args.num_max_people),
            overlap_merge_iou_thresh=float(args.overlap_merge_iou_thresh),
            overlap_merge_min_frames=5,
            pose_cos_thresh=float(args.pose_cos_thresh),
            pose_max_gap=int(args.pose_max_gap),
            pose_min_iou_for_pair=float(args.pose_min_iou_for_pair),
            pose_max_center_dist=float(args.pose_max_center_dist),
        )

        expected_per_frame = _load_gt_count_per_frame(clip.gt, n)

        out_path = output_dir / f"{clip.name}_tracking_overlay.mp4"
        cfg = RenderConfig(
            max_frames=args.max_frames,
            config_label=config_label,
        )
        try:
            summary = _render_clip_overlay(
                clip=clip,
                tracks=tracks,
                n_decoded_frames=n,
                output_path=out_path,
                cfg=cfg,
                expected_per_frame=expected_per_frame,
                expected_const=clip.expected_const,
            )
        except Exception as exc:
            log.exception("[%s] render failed: %s", clip.name, exc)
            continue

        summaries.append(summary)
        per_clip_json = output_dir / f"{clip.name}_summary.json"
        per_clip_json.write_text(json.dumps(summary, indent=2))

    index = output_dir / "index.json"
    index.write_text(json.dumps({
        "config": {
            "gaussian_sigma": args.gaussian_sigma,
            "min_total_frames": args.min_total_frames,
            "min_conf": args.min_conf,
            "max_gap_interp": args.max_gap_interp,
            "config_label": config_label,
        },
        "clips": summaries,
    }, indent=2))
    log.info("wrote %s with %d clip summaries", index, len(summaries))
    return 0 if summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
