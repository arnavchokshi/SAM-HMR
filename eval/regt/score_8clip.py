#!/usr/bin/env python3
"""Score one or more BoxMOT cache roots on the **8-clip** registry
(legacy 6 + loveTest + shorterTest), with IDF1/MOTA/IDsw/count_exact
per clip and a `mean6_idf1` aggregate over the 6 GT non-leaked clips.

Usage:
    python eval/regt/score_8clip.py                            # default winners
    python eval/regt/score_8clip.py --variant ocsort_ens_768_1024
    python eval/regt/score_8clip.py --post legacy_mtf24_noprox
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.overfit import _common as oc  # noqa: E402

POSTS = {
    "winner_mtf40_prox60": oc.PostCfg(
        min_total_frames=40, min_conf=0.38,
        pose_max_center_dist=60.0, pose_max_gap=40,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf30_prox60": oc.PostCfg(
        min_total_frames=30, min_conf=0.38,
        pose_max_center_dist=60.0, pose_max_gap=40,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "legacy_mtf24_noprox": oc.PostCfg(
        min_total_frames=24, min_conf=0.38,
        pose_max_center_dist=float("inf"), pose_max_gap=40,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf40_prox100": oc.PostCfg(
        min_total_frames=40, min_conf=0.38,
        pose_max_center_dist=100.0, pose_max_gap=80,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf40_prox150": oc.PostCfg(
        min_total_frames=40, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox100": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=100.0, pose_max_gap=80,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf40_prox100_idmerge75": oc.PostCfg(
        min_total_frames=40, min_conf=0.38,
        pose_max_center_dist=100.0, pose_max_gap=80,
        id_merge_iou_thresh=0.4, id_merge_max_gap=24,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox150": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox200": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=200.0, pose_max_gap=160,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf80_prox150": oc.PostCfg(
        min_total_frames=80, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox120_gap80": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=120.0, pose_max_gap=80,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf70_prox150": oc.PostCfg(
        min_total_frames=70, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf75_prox150": oc.PostCfg(
        min_total_frames=75, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf90_prox150": oc.PostCfg(
        min_total_frames=90, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox150_conf30": oc.PostCfg(
        min_total_frames=60, min_conf=0.30,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox150_conf45": oc.PostCfg(
        min_total_frames=60, min_conf=0.45,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.0, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox150_pose85": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.85, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox150_pose80": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.80, pose_min_iou_for_pair=0.0,
    ),
    "winner_mtf60_prox150_pose90": oc.PostCfg(
        min_total_frames=60, min_conf=0.38,
        pose_max_center_dist=150.0, pose_max_gap=120,
        pose_cos_thresh=0.90, pose_min_iou_for_pair=0.0,
    ),
}

# Non-leaked GT clips (mirrorTest is in YOLO fine-tune, exclude from mean).
CLIPS_NONLEAKED_GT = ("gymTest", "adiTest", "BigTest", "easyTest",
                      "loveTest", "shorterTest")


def score_variant(
    cache_dir: Path,
    post: oc.PostCfg,
    user_clips_root: "Path | None" = None,
) -> Dict[str, Dict[str, float]]:
    original = oc.CACHE_ROOT
    oc.CACHE_ROOT = cache_dir
    try:
        clips = oc.all_clips_with_cache(user_clips_root=user_clips_root)
        return oc.score_config_on_all_clips(
            post=post, clips=clips, enable_idf1=True,
        )
    finally:
        oc.CACHE_ROOT = original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=None,
                    help="bestofboth_gh200 subdirs to score; default = all")
    ap.add_argument("--cache-base", type=Path,
                    default=REPO_ROOT / "runs" / "bestofboth_gh200")
    ap.add_argument("--post", default="winner_mtf40_prox60",
                    choices=list(POSTS.keys()) + ["all"],
                    help="postprocess preset")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "runs" / "bestofboth_gh200" / "scores_8clip.json")
    ap.add_argument("--user-clips-root", type=Path, default=None,
                    help="Root for clip GT lookup (defaults to "
                         "/Users/arnavchokshi/Desktop). Use /work/data/clips "
                         "on the GH200.")
    args = ap.parse_args()

    if args.variants:
        variants = args.variants
    else:
        variants = sorted(p.name for p in args.cache_base.iterdir()
                          if (p / "_cache").is_dir())

    posts = POSTS if args.post == "all" else {args.post: POSTS[args.post]}

    rows = []
    for v in variants:
        cdir = args.cache_base / v / "_cache"
        if not cdir.is_dir():
            print(f"[skip] {v}: missing _cache")
            continue
        for plabel, post in posts.items():
            scores = score_variant(
                cdir, post, user_clips_root=args.user_clips_root,
            )
            mean_idf1: List[float] = []
            mean_mota: List[float] = []
            mean_count: List[float] = []
            row = {"variant": v, "post": plabel, "scores": {}}
            for c in ("mirrorTest",) + CLIPS_NONLEAKED_GT + ("2pplTest",):
                s = scores.get(c, {})
                row["scores"][c] = {
                    "idf1": s.get("idf1"),
                    "mota": s.get("mota"),
                    "idsw": s.get("idsw"),
                    "count_exact": s.get("count_exact_acc"),
                    "ids_pred": s.get("unique_ids_pred"),
                    "ids_gt": s.get("unique_ids_gt"),
                }
                if c in CLIPS_NONLEAKED_GT and s.get("idf1") is not None:
                    mean_idf1.append(float(s["idf1"]))
                    mean_mota.append(float(s["mota"]))
                    mean_count.append(float(s["count_exact_acc"]))
            row["mean6nl_idf1"] = (sum(mean_idf1) / len(mean_idf1)) if mean_idf1 else None
            row["mean6nl_mota"] = (sum(mean_mota) / len(mean_mota)) if mean_mota else None
            row["mean6nl_count_exact"] = (sum(mean_count) / len(mean_count)) if mean_count else None
            rows.append(row)
            sc = row["scores"]
            print(
                f"\n=== {v}  |  {plabel} ===\n"
                f"  BigTest    IDF1={_fmt(sc['BigTest']['idf1'])} mota={_fmt(sc['BigTest']['mota'])} "
                f"ce={_fmt(sc['BigTest']['count_exact'])} ids={sc['BigTest']['ids_pred']}/{sc['BigTest']['ids_gt']}\n"
                f"  mirror     IDF1={_fmt(sc['mirrorTest']['idf1'])} mota={_fmt(sc['mirrorTest']['mota'])} "
                f"ce={_fmt(sc['mirrorTest']['count_exact'])} ids={sc['mirrorTest']['ids_pred']}/{sc['mirrorTest']['ids_gt']}  (LEAKED)\n"
                f"  gym        IDF1={_fmt(sc['gymTest']['idf1'])} mota={_fmt(sc['gymTest']['mota'])} "
                f"ce={_fmt(sc['gymTest']['count_exact'])} ids={sc['gymTest']['ids_pred']}/{sc['gymTest']['ids_gt']}\n"
                f"  adi        IDF1={_fmt(sc['adiTest']['idf1'])} mota={_fmt(sc['adiTest']['mota'])} "
                f"ce={_fmt(sc['adiTest']['count_exact'])} ids={sc['adiTest']['ids_pred']}/{sc['adiTest']['ids_gt']}\n"
                f"  easy       IDF1={_fmt(sc['easyTest']['idf1'])} mota={_fmt(sc['easyTest']['mota'])} "
                f"ce={_fmt(sc['easyTest']['count_exact'])} ids={sc['easyTest']['ids_pred']}/{sc['easyTest']['ids_gt']}\n"
                f"  loveTest   IDF1={_fmt(sc['loveTest']['idf1'])} mota={_fmt(sc['loveTest']['mota'])} "
                f"ce={_fmt(sc['loveTest']['count_exact'])} ids={sc['loveTest']['ids_pred']}/{sc['loveTest']['ids_gt']}\n"
                f"  shorterTest IDF1={_fmt(sc['shorterTest']['idf1'])} mota={_fmt(sc['shorterTest']['mota'])} "
                f"ce={_fmt(sc['shorterTest']['count_exact'])} ids={sc['shorterTest']['ids_pred']}/{sc['shorterTest']['ids_gt']}\n"
                f"  2ppl       IDF1=  n/a  mota=  n/a  ce={_fmt(sc['2pplTest']['count_exact'])} ids={sc['2pplTest']['ids_pred']}/{sc['2pplTest']['ids_gt']}\n"
                f"  mean6nl_IDF1={_fmt(row['mean6nl_idf1'], 4)}  "
                f"mean6nl_MOTA={_fmt(row['mean6nl_mota'], 4)}  "
                f"mean6nl_count_exact={_fmt(row['mean6nl_count_exact'], 4)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.output}")
    return 0


def _fmt(v, prec: int = 3) -> str:
    if v is None:
        return "  n/a"
    try:
        return f"{float(v):.{prec}f}"
    except Exception:
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
