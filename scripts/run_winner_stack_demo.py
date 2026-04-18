#!/usr/bin/env python3
"""Run the shipped DeepOcSort tracking pipeline on the canonical clips.

This is THE one-shot entry point: detector ensemble + DeepOcSort +
post-processing + overlay rendering, with the winner config locked in.
The same path that produced the published 8-clip scoreboard
(mean6nl IDF1 = 0.949 across the 6 non-leaked GT clips).

Output layout::

    runs/winner_stack_demo/
      _cache/<clip>/*.pkl                  # per-clip FrameDetections cache
      overlays/<clip>_tracking_overlay.mp4 # ID overlays (optional)
      timings.json

Reproduce a single clip from another script with::

    python eval/run_boxmot_tracker.py \\
        --tracker DeepOcSort \\
        --imgsz-ensemble 768 1024 --ensemble-iou 0.6 \\
        --conf 0.31 --iou 0.70 --device <auto> \\
        --output runs/<your_run>

See docs/WINNING_PIPELINE_CONFIGURATION.md for the full config and
docs/EXPERIMENTS_LOG.md for why DeepOcSort + ensemble [768, 1024] is the
shipped default.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("run_winner_stack_demo")


# Winner detector + tracker config (locked).
WINNER_TRACKER = "DeepOcSort"
WINNER_IMGSZ_ENSEMBLE = (768, 1024)
WINNER_ENSEMBLE_IOU = 0.6
WINNER_CONF = 0.31
WINNER_IOU = 0.70
WINNER_REID = "osnet_x0_25_msmt17.pt"

# Winner postprocess (used when rendering overlays). Match
# docs/WINNING_PIPELINE_CONFIGURATION.md §1.2.
WINNER_MIN_TOTAL_FRAMES = 60
WINNER_MIN_CONF = 0.38
WINNER_POSE_MAX_CENTER_DIST = 150
WINNER_POSE_MAX_GAP = 120


def _pick_device(prefer: str) -> str:
    import torch

    prefer = prefer.strip().lower()
    if prefer in ("auto", ""):
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if prefer in ("cuda", "cuda:0", "gpu") and torch.cuda.is_available():
        return "cuda:0"
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cpu":
        return "cpu"
    log.warning("device %r not available; falling back to auto", prefer)
    return _pick_device("auto")


def _jobs(desktop: Path) -> List[Tuple[str, Path, bool, Optional[int]]]:
    """(clip_name, video_path, leaked, max_frames) — the 8-clip canonical set."""
    return [
        ("mirrorTest", desktop / "mirrorTest" / "IMG_2946.MP4", True, None),
        ("gymTest", desktop / "gymTest" / "IMG_8309.mov", False, None),
        ("adiTest", desktop / "adiTest" / "IMG_1649.mov", False, 188),
        ("BigTest", desktop / "BigTest" / "BigTest.mov", False, None),
        ("easyTest", desktop / "easyTest" / "IMG_2082.mov", False, None),
        ("2pplTest", desktop / "v12044gd0000d798o7nog65hc803k98g.mov", False, None),
        ("loveTest", desktop / "loveTest" / "IMG_9265.mov", False, None),
        ("shorterTest", desktop / "shorterTest" / "TestVideo.mov", False, None),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--desktop", type=Path,
                   default=Path("/Users/arnavchokshi/Desktop"),
                   help="Folder containing the canonical clips (default: ~/Desktop).")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "runs" / "winner_stack_demo")
    p.add_argument("--device", default="auto",
                   help="auto | cuda | mps | cpu")
    p.add_argument("--clips", nargs="*", default=None,
                   help="Restrict to a subset of canonical clip names.")
    p.add_argument("--skip-tracking", action="store_true",
                   help="Only render overlays from existing _cache.")
    p.add_argument("--skip-render", action="store_true",
                   help="Only run tracking; skip overlay MP4s.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from eval.run_boxmot_tracker import (  # noqa: E402
        BoxmotConfig,
        ClipSpec,
        run_one_clip,
    )

    desktop = args.desktop.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()
    cache_root = out_dir / "_cache"
    overlay_dir = out_dir / "overlays"
    cache_root.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(args.device)
    log.info("device=%s, output=%s", device, out_dir)

    weights = REPO_ROOT / "weights" / "best.pt"
    if not weights.is_file():
        log.error("missing YOLO weights: %s", weights)
        return 1

    cfg = BoxmotConfig(
        tracker_name=WINNER_TRACKER,
        reid_weights=Path(WINNER_REID),
        imgsz=WINNER_IMGSZ_ENSEMBLE[0],
        imgsz_ensemble=WINNER_IMGSZ_ENSEMBLE,
        ensemble_iou=WINNER_ENSEMBLE_IOU,
        conf=WINNER_CONF,
        iou=WINNER_IOU,
        device=device,
        half=False,
    )
    log.info("tracker config: %s", cfg)

    jobs = _jobs(desktop)
    if args.clips:
        wanted = set(args.clips)
        jobs = [j for j in jobs if j[0] in wanted]

    timings = {"device": device, "weights": str(weights), "clips": {}}

    for name, video, leaked, max_frames in jobs:
        if not video.is_file():
            log.warning("[%s] video not found at %s — skipping", name, video)
            continue
        spec = ClipSpec(name=name, video=video, leaked=leaked, max_frames=max_frames)

        if not args.skip_tracking:
            log.info("[%s] tracking ...", name)
            t0 = time.perf_counter()
            run_one_clip(clip=spec, cfg=cfg, weights=weights,
                         cache_root=cache_root, force=True)
            elapsed = time.perf_counter() - t0
            timings["clips"][name] = {"seconds": round(elapsed, 3),
                                      "video": str(video)}
            log.info("[%s] done in %.1fs", name, elapsed)

    timings_path = out_dir / "timings.json"
    timings_path.write_text(json.dumps(timings, indent=2))
    log.info("wrote %s", timings_path)

    if args.skip_render:
        return 0

    # Always pass an explicit override for the 2ppl video so the renderer
    # finds the same file the tracker just consumed.
    extra_clip_args: List[str] = []
    for name, video, _leaked, _mf in jobs:
        extra_clip_args.extend(["--clip-video", f"{name}={video}"])

    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "render_overlay_videos.py"),
        "--cache-root", str(cache_root),
        "--user-clips-root", str(desktop),
        "--output", str(overlay_dir),
        "--min-total-frames", str(WINNER_MIN_TOTAL_FRAMES),
        "--min-conf", str(WINNER_MIN_CONF),
        "--pose-max-center-dist", str(WINNER_POSE_MAX_CENTER_DIST),
        "--pose-max-gap", str(WINNER_POSE_MAX_GAP),
        "--config-label",
        "DeepOcSort | ens 768+1024 | mtf=60 prox=150 gap=120",
        *extra_clip_args,
    ]
    if args.clips:
        cmd.extend(["--clips", *args.clips])

    rc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if rc.returncode != 0:
        log.error("render_overlay_videos.py failed (%s)", rc.returncode)
        return int(rc.returncode)
    log.info("overlays in %s", overlay_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
