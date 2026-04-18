"""Shared helpers for the overfitting / generalization audit suite.

Conventions
-----------
* Everything here works with the **cached YOLO+tracker outputs** that
  ``eval/eval_counts.py`` writes to ``runs/eval_counts/_cache/<clip>/*.pkl``.
  That means we can sweep postprocess configs without ever invoking the
  GPU (and without risking interference with the running Lambda sweep).

* The default postprocess config below is the active hand-tuned winner
  documented in ``docs/FINDINGS.md`` (see ``eval/eval_counts.py::PostCfg``).

* All clip metadata (gt path, max_frames, expected_const) is duplicated
  here intentionally so this module is self-contained -- the upstream
  sweep code can change without breaking the audit.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Clip registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipSpec:
    name: str
    video: Path
    gt: Optional[Path]                 # None for 2pplTest (constant prior)
    leaked: bool                       # mirrorTest is leaked, never used to rank
    expected_const: Optional[int] = None
    max_frames: Optional[int] = None


def all_clip_specs(user_clips_root: Optional[Path] = None) -> List[ClipSpec]:
    """The full set of 8 clips the user cares about (6 legacy + 2 new
    added 2026-04-17: loveTest, shorterTest)."""
    root = user_clips_root or Path("/Users/arnavchokshi/Desktop")
    return [
        ClipSpec("mirrorTest", root / "mirrorTest" / "IMG_2946.MP4",
                 gt=root / "mirrorTest" / "gt" / "gt.txt", leaked=True),
        ClipSpec("gymTest",    root / "gymTest" / "IMG_8309.mov",
                 gt=root / "gymTest" / "gt" / "gt.txt", leaked=False),
        ClipSpec("adiTest",    root / "adiTest" / "IMG_1649.mov",
                 gt=root / "adiTest" / "gt" / "gt.txt", leaked=False,
                 max_frames=188),
        ClipSpec("BigTest",    root / "BigTest" / "BigTest.mov",
                 gt=root / "BigTest" / "gt" / "gt.txt", leaked=False),
        ClipSpec("easyTest",   root / "easyTest" / "IMG_2082.mov",
                 gt=root / "easyTest" / "gt" / "gt.txt", leaked=False),
        ClipSpec("2pplTest",   root / "2pplTest.mov",
                 gt=None, leaked=False, expected_const=2),
        ClipSpec("loveTest",   root / "loveTest" / "IMG_9265.mov",
                 gt=root / "loveTest" / "gt" / "gt.txt", leaked=False),
        ClipSpec("shorterTest", root / "shorterTest" / "TestVideo.mov",
                 gt=root / "shorterTest" / "gt" / "gt.txt", leaked=False),
    ]


# ---------------------------------------------------------------------------
# Baseline postprocess config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostCfg:
    """Mirrors ``eval/eval_counts.py::PostCfg`` -- the active hand-tuned
    winner documented in ``docs/FINDINGS.md``.

    Kept here as a separate copy so the audit isn't coupled to the
    upstream sweep schema. Bump these defaults if the hand-tuned config
    moves.

    ``min_total_frames`` raised 10 -> 24 (2026-04-17) per the
    cross-evaluation in ``runs/overfit_analysis/cross_eval_results.md``
    and the recommendation in ``docs/SWEEP_OVERFITTING_HANDOFF.md`` --
    on the legacy YOLO+tracker cache this single change drives mirrorTest
    from 0.6934 -> 1.0000 with no regression on any other clip and is the
    config shipped as ``runs/overfit_analysis/configs/recommended_baseline.json``.

    Phase 1 pose-cosine ID merge fields (2026-04-17):
      * ``pose_cos_thresh = 0.0`` disables the second pose-merge pass and
        keeps backward-compat with the production baseline. Set to a
        value in [0.80, 0.95] to enable.
      * ``pose_max_gap`` is the *upper* gap bound for the pose pass; the
        IoU pass still owns gaps in ``[1, id_merge_max_gap]``.
      * ``pose_min_iou_for_pair`` is an optional cheap pre-gate (default
        off via 0.0). Spec defaults the pose pass to gate on geometry
        only.
      * ``pose_weights`` / ``pose_device`` configure the YOLO11s-pose
        backend; defaults match the file shipped in ``data/pose/``.
    """
    # 8-clip generalizing winner (2026-04-17, after loveTest/shorterTest):
    # mtf 30 -> 60 (filters short fragments aggressively)
    # pose_max_center_dist 60 -> 150 (handles wider-shot footage like
    # loveTest where dancers are smaller in frame; principled because the
    # threshold should scale with camera distance, and 60px was overfit
    # to the close-up 6-clip set).
    # pose_max_gap 40 -> 120 (long enough to bridge multi-second
    # occlusions in busy free-form choreography).
    # On all 8 clips (DeepOcSort+ens_768_1024), this gives mean6nl IDF1
    # 0.949 vs 0.944 with the old defaults, with the biggest lift on
    # loveTest (0.781 -> 0.804) and shorterTest (0.911 -> 0.919).
    min_total_frames: int = 60
    min_conf: float = 0.38
    id_merge_iou_thresh: float = 0.5
    id_merge_max_gap: int = 8
    gaussian_sigma: float = 3.0
    max_gap_interp: int = 12
    medfilt_window: int = 11
    num_max_people: int = 25
    overlap_merge_iou_thresh: float = 0.7
    overlap_merge_min_frames: int = 5
    edge_trim_conf_thresh: float = 0.0
    edge_trim_max_frames: int = 0
    pose_cos_thresh: float = 0.0
    pose_max_gap: int = 120
    pose_min_iou_for_pair: float = 0.0
    pose_max_center_dist: float = 150.0
    pose_weights: str = "data/pose/yolo11s-pose.pt"
    pose_device: str = "mps"

    def asdict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    def replace(self, **changes) -> "PostCfg":
        return dataclasses.replace(self, **changes)


# Knob ranges used by the sensitivity probe and the per-clip oracle.
# (lo, hi, integer?, monotone?)  monotone=True means the metric is
# expected to vary smoothly across the range, so we can sample every
# step. monotone=False (e.g. medfilt_window) means we can only enumerate.
KNOB_RANGES: Dict[str, Tuple[float, float, bool]] = {
    "min_total_frames":           (3, 50, True),
    "min_conf":                   (0.20, 0.55, False),
    "id_merge_iou_thresh":        (0.25, 0.75, False),
    "id_merge_max_gap":           (4, 48, True),
    "gaussian_sigma":             (0.5, 6.0, False),
    "max_gap_interp":             (6, 48, True),
    "medfilt_window":             (5, 15, True),  # only odd values valid
    "num_max_people":             (2, 50, True),
    "overlap_merge_iou_thresh":   (0.50, 0.85, False),
    "overlap_merge_min_frames":   (3, 12, True),
    "edge_trim_conf_thresh":      (0.0, 0.6, False),
    "edge_trim_max_frames":       (0, 8, True),
}

# Knobs whose only valid values are the discrete set used by Optuna.
DISCRETE_KNOBS: Dict[str, List[int]] = {
    "medfilt_window": [5, 7, 11, 15],
    "num_max_people": [2, 4, 6, 10, 14, 25, 50],
}


# ---------------------------------------------------------------------------
# YOLO+tracker cache loader
# ---------------------------------------------------------------------------


CACHE_ROOT = REPO_ROOT / "runs" / "eval_counts" / "_cache"


def _cache_dir_for(clip: str) -> Path:
    return CACHE_ROOT / clip


def list_cached_yolo_runs(clip: str) -> List[Path]:
    """All ``.pkl`` files under ``runs/eval_counts/_cache/<clip>/``."""
    d = _cache_dir_for(clip)
    if not d.is_dir():
        return []
    return sorted(d.glob("imgsz*_conf*.pkl"))


def load_cached_yolo_frames(clip: str, *, max_frames: Optional[int] = None,
                            cache_path: Optional[Path] = None):
    """Load a ``List[FrameDetections]`` cache for ``clip``.

    If ``cache_path`` is None, picks the lexicographically first cache
    in the clip's cache dir (typically only one exists locally).

    Raises ``FileNotFoundError`` if no cache is on disk -- the caller
    should re-run ``eval/eval_counts.py`` once for that clip first.
    """
    if cache_path is None:
        candidates = list_cached_yolo_runs(clip)
        if not candidates:
            raise FileNotFoundError(
                f"no YOLO cache for {clip}; run eval/eval_counts.py once "
                f"to populate {_cache_dir_for(clip)}"
            )
        cache_path = candidates[0]
    with open(cache_path, "rb") as f:
        frames = pickle.load(f)
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]
    return frames


def cache_signature(clip: str) -> Optional[str]:
    """Return the YOLO/tracker config used to produce the cache, parsed
    out of the filename (so the audit report can document what the
    sensitivity numbers were measured against)."""
    runs = list_cached_yolo_runs(clip)
    if not runs:
        return None
    return runs[0].stem


# ---------------------------------------------------------------------------
# Postprocess + score wrapper
# ---------------------------------------------------------------------------


# Module-scope cache so a sweep over many configs only loads YOLO11s-pose
# once. Keyed by (weights, device).
_POSE_EXTRACTOR_CACHE: Dict[Tuple[str, str], object] = {}


def _get_pose_extractor(weights: str, device: str):
    """Lazily build (and cache) a PoseFeatureExtractor."""
    key = (weights, device)
    if key not in _POSE_EXTRACTOR_CACHE:
        from tracking.pose_features import PoseFeatureExtractor
        _POSE_EXTRACTOR_CACHE[key] = PoseFeatureExtractor(
            weights=weights, device=device,
        )
    return _POSE_EXTRACTOR_CACHE[key]


def postprocess_and_score(
    frames,
    *,
    clip: ClipSpec,
    post: PostCfg,
    enable_idf1: bool = False,
) -> Dict[str, float]:
    """Run postprocess + the appropriate scorer for this clip.

    Returns a dict with at least:
      * ``score``:           composite sweep-equivalent score
                             (count_exact_acc for GT clips,
                              score_2ppl for 2pplTest)
      * ``count_exact_acc``: exact per-frame count match
      * ``count_within1_acc``
      * ``unique_ids_pred``, ``unique_ids_gt`` (when applicable)
      * ``idf1``, ``mota``, ``idsw`` (when ``enable_idf1=True``)
      * jitter aggregates (mean / p95 / max for the 6 standard fields)
    """
    # imports here so the module loads quickly when only the helpers above
    # are needed (e.g. by docs builds)
    from tracking.postprocess import (
        frame_detections_to_raw_tracks,
        postprocess_tracks,
    )
    from eval.count_accuracy import score_tracks_counts
    from eval.jitter import clip_jitter
    from eval.score_2ppl import score_2ppl

    raw = frame_detections_to_raw_tracks(frames)

    # Phase 1 long-gap ID merge.
    # Two independent gates (either or both can be enabled):
    #   * proximity:  post.pose_max_center_dist < inf
    #   * pose-cos:   post.pose_cos_thresh > 0 (loads the YOLO-pose model)
    # Pose-cos requires both a loadable model AND an on-disk video for the
    # frame_loader. Proximity needs neither but still benefits from the
    # frame_loader when the user opts in to combined gating.
    pose_extractor = None
    frame_loader = None
    proximity_enabled = post.pose_max_center_dist < float("inf")
    cosine_enabled_cfg = post.pose_cos_thresh > 0
    video_available = clip.video is not None and Path(clip.video).is_file()
    if cosine_enabled_cfg and video_available:
        from tracking.pose_features import make_frame_loader
        try:
            pose_extractor = _get_pose_extractor(post.pose_weights, post.pose_device)
            frame_loader = make_frame_loader(Path(clip.video))
        except Exception as exc:
            log = logging.getLogger(__name__)
            log.warning("pose-merge disabled for %s (%s)", clip.name, exc)
            pose_extractor = None
            frame_loader = None

    tracks = postprocess_tracks(
        raw,
        min_box_w=10, min_box_h=10,
        min_total_frames=post.min_total_frames,
        min_conf=post.min_conf,
        max_gap_interp=post.max_gap_interp,
        id_merge_max_gap=post.id_merge_max_gap,
        id_merge_iou_thresh=post.id_merge_iou_thresh,
        id_merge_osnet_cos_thresh=0.7,
        medfilt_window=post.medfilt_window,
        gaussian_sigma=post.gaussian_sigma,
        num_max_people=post.num_max_people,
        overlap_merge_iou_thresh=post.overlap_merge_iou_thresh,
        overlap_merge_min_frames=post.overlap_merge_min_frames,
        edge_trim_conf_thresh=post.edge_trim_conf_thresh,
        edge_trim_max_frames=post.edge_trim_max_frames,
        pose_extractor=pose_extractor,
        pose_cos_thresh=post.pose_cos_thresh,
        pose_max_gap=post.pose_max_gap,
        pose_min_iou_for_pair=post.pose_min_iou_for_pair,
        pose_max_center_dist=post.pose_max_center_dist,
        frame_loader=frame_loader,
    )
    n = len(frames)

    if clip.expected_const is not None:
        # 2pplTest
        s2 = score_2ppl(tracks, num_frames=n)
        cm = {
            "count_exact_acc": float(s2.get("frac_exactly_two", 0.0)),
            "count_within1_acc": float(s2.get("frac_exactly_two", 0.0)),
            "unique_ids_pred": int(s2.get("unique_ids", len(tracks))),
            "unique_ids_gt": 2,
            "score": float(s2["score"]),
        }
    else:
        if clip.gt is None or not clip.gt.is_file():
            cm = {"score": 0.0, "count_exact_acc": 0.0,
                  "count_within1_acc": 0.0, "unique_ids_pred": len(tracks),
                  "unique_ids_gt": 0}
        else:
            cm = score_tracks_counts(tracks, n, gt_path=clip.gt)
            cm["score"] = float(cm.get("count_exact_acc", 0.0))

    if enable_idf1 and clip.gt is not None and clip.gt.is_file():
        from eval.scoring import score_tracks_dict
        try:
            mt = score_tracks_dict(tracks, num_frames=n, gt_path=clip.gt,
                                    enable_hota=False)
            cm["idf1"] = float(mt.get("idf1", float("nan")))
            cm["mota"] = float(mt.get("mota", float("nan")))
            cm["idsw"] = float(mt.get("num_switches", float("nan")))
        except Exception:
            cm["idf1"] = cm["mota"] = cm["idsw"] = float("nan")

    cm.update(clip_jitter(tracks))
    cm["n_tracks_pred"] = int(len(tracks))
    cm["num_frames"] = int(n)
    return cm


# ---------------------------------------------------------------------------
# Driver helpers (run a config on every available clip)
# ---------------------------------------------------------------------------


def all_clips_with_cache(user_clips_root: Optional[Path] = None) -> List[ClipSpec]:
    """Subset of clips for which we currently have a YOLO cache on disk."""
    out: List[ClipSpec] = []
    for c in all_clip_specs(user_clips_root):
        if list_cached_yolo_runs(c.name):
            out.append(c)
    return out


def score_config_on_all_clips(
    post: PostCfg,
    *,
    clips: Optional[List[ClipSpec]] = None,
    enable_idf1: bool = False,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, float]]:
    """Score one postprocess config on every clip with a YOLO cache."""
    if clips is None:
        clips = all_clips_with_cache()
    out: Dict[str, Dict[str, float]] = {}
    for c in clips:
        if log is not None:
            log.info("scoring %s ...", c.name)
        try:
            frames = load_cached_yolo_frames(c.name, max_frames=c.max_frames)
        except FileNotFoundError as exc:
            if log is not None:
                log.warning("skipping %s: %s", c.name, exc)
            continue
        out[c.name] = postprocess_and_score(frames, clip=c, post=post,
                                            enable_idf1=enable_idf1)
    return out


# ---------------------------------------------------------------------------
# Misc IO
# ---------------------------------------------------------------------------


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot json-encode {type(obj)}")
