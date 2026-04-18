"""Stage A — produce frames + DeepOcSort tracks for one clip.

Runs in the existing tracking conda env. No HMR-specific deps.

Usage:
    python -m threed.stage_a.run_stage_a --clip adiTest \
        --video /Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov \
        --cache-dir runs/winner_stack_demo/_cache/adiTest
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from threed.config import default_config
from threed.io import save_tracks
from threed.stage_a.extract_tracks import extract_tracks_from_cache
from threed.stage_a.extract_frames import extract_frames

log = logging.getLogger("stage_a")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Path to runs/<run>/_cache/<clip> with one .pkl inside")
    p.add_argument("--min-total-frames", type=int, default=60)
    p.add_argument("--min-conf", type=float, default=0.38)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = default_config()
    dirs = cfg.clip_dirs(args.clip).ensure()

    log.info("[%s] extracting frames from %s", args.clip, args.video)
    n_frames = extract_frames(
        args.video,
        out_dir_resized=dirs.intermediates / "frames",
        out_dir_full=dirs.intermediates / "frames_full",
        max_height=cfg.max_height,
    )
    log.info("[%s] wrote %d frames", args.clip, n_frames)

    log.info("[%s] extracting tracks from %s", args.clip, args.cache_dir)
    tracks = extract_tracks_from_cache(
        args.cache_dir,
        min_total_frames=args.min_total_frames,
        min_conf=args.min_conf,
    )
    log.info("[%s] %d tracks survived pruning", args.clip, len(tracks))

    save_tracks(tracks, dirs.intermediates / "tracks.pkl")
    log.info("[%s] wrote %s", args.clip, dirs.intermediates / "tracks.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
