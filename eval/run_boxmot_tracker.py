"""Run a BoxMOT tracker (DeepOCSort, BotSort, OcSort, ...) over our 6
canonical clips and write the results in the same ``FrameDetections``
cache layout that ``eval/eval_counts.py`` produces.

This is the integration glue called for in
``docs/SWEEP_OVERFITTING_HANDOFF.md`` §5d step 2: swap the tracker
without touching the detector or the postprocess, then re-score with
the existing audit pipeline so the comparison is apples-to-apples.

Cache layout produced::

    runs/tracker_experiments/boxmot_<tracker>/_cache/<clip>/
        <hash>.pkl               # list[FrameDetections] (legacy schema)

The cache directory is consumable by
``eval/overfit/audit_with_cache.py --cache-root <dir>`` and by
``eval/overfit/_common.score_config_on_all_clips`` after pointing
``_common.CACHE_ROOT`` at it.

Usage::

    # DeepOCSort with default OSNet ReID (auto-downloaded)
    python eval/run_boxmot_tracker.py \\
        --tracker DeepOcSort \\
        --output  runs/tracker_experiments/deepocsort

    # Same, but only run the two cheap clips first to validate
    python eval/run_boxmot_tracker.py \\
        --tracker DeepOcSort --clips adiTest 2pplTest \\
        --output  runs/tracker_experiments/deepocsort
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prune_tracks import FrameDetections  # noqa: E402

log = logging.getLogger("run_boxmot_tracker")


# --------------------------------------------------------------------------
# Patch BoxMOT's Kalman update with cholesky jitter
# --------------------------------------------------------------------------
#
# DeepOcSort (and any BoxMOT tracker that inherits from BaseKalmanFilter)
# occasionally produces a covariance matrix that is barely positive
# semi-definite -- ``scipy.linalg.cho_factor`` then raises
# ``LinAlgError`` and the whole frame's track update is aborted.
#
# In our smoke test on mirrorTest this happened on ~25% of frames,
# silently dropping every track on those frames and grossly under-
# scoring the tracker. The standard textbook fix is "cholesky jitter":
# add ``epsilon * I`` to the projected covariance, retry, double
# epsilon, repeat. We patch it once at runner startup so every BoxMOT
# tracker we test inherits the fix.

_JITTER_PATCH_INSTALLED = False


def _install_kalman_jitter_patch() -> None:
    global _JITTER_PATCH_INSTALLED
    if _JITTER_PATCH_INSTALLED:
        return
    import scipy.linalg
    # boxmot has shuffled its kalman-filter module path across versions.
    # Try the legacy path first, then the >=10.0.52 layout. If neither
    # exists, the patch is a no-op (we still get correct behaviour, just
    # without the jitter retry — current OcSort uses its own internal
    # kf and rarely raises).
    try:
        from boxmot.motion.kalman_filters.base import BaseKalmanFilter  # type: ignore
    except ImportError:
        try:
            from boxmot.motion.kalman_filters.xysr_kf import (  # type: ignore
                KalmanFilterXYSR as BaseKalmanFilter,
            )
        except ImportError:
            log.info("kalman jitter patch skipped: boxmot has no recognized "
                     "BaseKalmanFilter (%s)", "")
            _JITTER_PATCH_INSTALLED = True
            return

    if not hasattr(BaseKalmanFilter, "update_state"):
        # In >=10.0.52 the equivalent method is just `update`.
        if hasattr(BaseKalmanFilter, "update"):
            _orig_update_state = BaseKalmanFilter.update

            def update_with_jitter(self, z, R=None, H=None,
                                   _max_tries: int = 6, _eps0: float = 1e-6):
                last_exc: Optional[Exception] = None
                for attempt in range(_max_tries):
                    try:
                        return _orig_update_state(self, z)
                    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as exc:
                        last_exc = exc
                        eps = _eps0 * (10 ** attempt)
                        try:
                            dim = self.P.shape[0]
                            self.P = self.P + np.eye(dim) * eps
                        except Exception:
                            break
                log.debug("kf update jitter retries exhausted: %s", last_exc)

            BaseKalmanFilter.update = update_with_jitter
            _JITTER_PATCH_INSTALLED = True
            log.info("installed BoxMOT KalmanFilter cholesky-jitter patch (update)")
            return
        log.info("kalman jitter patch skipped: no compatible update method")
        _JITTER_PATCH_INSTALLED = True
        return

    _orig_update_state = BaseKalmanFilter.update_state

    def update_state_with_jitter(self, z, R=None, H=None,
                                 _max_tries: int = 6, _eps0: float = 1e-6):
        last_exc: Optional[Exception] = None
        for attempt in range(_max_tries):
            try:
                return _orig_update_state(self, z, R=R, H=H)
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as exc:
                last_exc = exc
                # Inflate the covariance diagonal and retry. This nudges
                # the projected covariance back into the SPD cone.
                eps = _eps0 * (10 ** attempt)
                try:
                    dim = self.P.shape[0]
                    self.P = self.P + np.eye(dim) * eps
                except Exception:
                    break
        log.debug(
            "BaseKalmanFilter.update_state: cholesky failed after %d jitter "
            "retries; skipping measurement (last err=%s)", _max_tries, last_exc,
        )

    BaseKalmanFilter.update_state = update_state_with_jitter
    _JITTER_PATCH_INSTALLED = True
    log.info("installed BoxMOT KalmanFilter cholesky-jitter patch")


DEFAULT_USER_CLIPS_ROOT = Path("/Users/arnavchokshi/Desktop")


@dataclass(frozen=True)
class ClipSpec:
    name: str
    video: Path
    leaked: bool
    max_frames: Optional[int] = None


def _all_clip_specs(root: Path) -> List[ClipSpec]:
    """Mirrors ``eval/eval_counts.py::_all_clip_specs``.

    8-clip set as of 2026-04-17 (added loveTest, shorterTest).
    """
    return [
        ClipSpec("mirrorTest", root / "mirrorTest" / "IMG_2946.MP4", leaked=True),
        ClipSpec("gymTest",    root / "gymTest" / "IMG_8309.mov",   leaked=False),
        ClipSpec("adiTest",    root / "adiTest" / "IMG_1649.mov",   leaked=False,
                 max_frames=188),
        ClipSpec("BigTest",    root / "BigTest" / "BigTest.mov",    leaked=False),
        ClipSpec("easyTest",   root / "easyTest" / "IMG_2082.mov",  leaked=False),
        ClipSpec("2pplTest",   root / "2pplTest.mov",               leaked=False),
        ClipSpec("loveTest",   root / "loveTest" / "IMG_9265.mov",  leaked=False),
        ClipSpec("shorterTest", root / "shorterTest" / "TestVideo.mov", leaked=False),
    ]


# --------------------------------------------------------------------------
# YOLO detector wrapper
# --------------------------------------------------------------------------

def _make_yolo_detector(weights: Path, *, imgsz: int, conf: float, iou: float,
                       device: str,
                       imgsz_ensemble: Optional[List[int]] = None,
                       ensemble_iou: float = 0.6,
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``frame_bgr -> np.ndarray[N, 6]`` of detections in
    BoxMOT layout ``[x1, y1, x2, y2, conf, cls]``.

    When ``imgsz_ensemble`` is provided (Phase 2 multi-scale), it
    overrides ``imgsz``: detection runs at every entry in the ensemble
    and the per-frame boxes are NMS-unioned at ``ensemble_iou`` (default
    0.6). Single-scale callers can keep passing ``imgsz`` and leave
    ``imgsz_ensemble`` as None.
    """
    from tracking.multi_scale_detector import make_multi_scale_detector

    if imgsz_ensemble:
        scales = list(imgsz_ensemble)
    else:
        scales = [int(imgsz)]

    return make_multi_scale_detector(
        Path(weights),
        imgsz_list=scales,
        conf=float(conf),
        iou=float(iou),
        device=str(device),
        ensemble_iou=float(ensemble_iou),
        classes=[0],
    )


# --------------------------------------------------------------------------
# BoxMOT tracker factory
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxmotConfig:
    tracker_name: str               # 'DeepOcSort', 'BotSort', 'OcSort', ...
    reid_weights: Optional[Path]    # None for trackers without ReID
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    imgsz: int = 768
    conf: float = 0.31
    iou: float = 0.70
    device: str = "mps"
    half: bool = False
    # Phase 2: multi-scale detection ensemble. None -> single-scale
    # behaviour (legacy). When set, overrides ``imgsz`` and runs YOLO
    # at every ``imgsz`` in the tuple, NMS-unioning the per-frame
    # detections at ``ensemble_iou``. Cache hash includes the tuple so
    # different scale combos do not collide.
    imgsz_ensemble: Optional[Tuple[int, ...]] = None
    ensemble_iou: float = 0.6

    def hash(self) -> str:
        """Stable hash for cache filenames."""
        payload = {
            "tracker_name": self.tracker_name,
            "reid_weights":
                str(self.reid_weights) if self.reid_weights else None,
            "extra_kwargs": self.extra_kwargs,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz_ensemble": (list(self.imgsz_ensemble)
                               if self.imgsz_ensemble else None),
            "ensemble_iou": (self.ensemble_iou
                             if self.imgsz_ensemble else None),
            # device / half intentionally not included -- they should not
            # change the tracking output (bit-for-bit modulo float noise).
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]


def _make_boxmot_tracker(cfg: BoxmotConfig):
    """Construct a BoxMOT tracker instance from the config."""
    _install_kalman_jitter_patch()
    import torch
    import boxmot as bm

    # Map our canonical short names to whatever the installed boxmot
    # actually exports. boxmot has shuffled its public class names
    # across versions (OcSort -> OCSort -> OCSORT etc.), so we look up
    # the requested name against a list of aliases and fall back to
    # the top-level boxmot module if the legacy ``boxmot.trackers``
    # namespace is empty (true for >=10.0.52).
    _aliases: Dict[str, List[str]] = {
        "OcSort":      ["OcSort", "OCSort", "OCSORT"],
        "DeepOcSort":  ["DeepOcSort", "DeepOCSort", "DeepOCSORT"],
        "BotSort":     ["BotSort", "BoTSort", "BoTSORT"],
        "ByteTrack":   ["ByteTrack", "BYTETracker", "BYTETrack"],
        "StrongSort":  ["StrongSort", "StrongSORT"],
        "HybridSort":  ["HybridSort", "HybridSORT"],
        "BoostTrack":  ["BoostTrack", "BoostTRACK"],
        "ImprAssoc":   ["ImprAssoc", "ImprAssocTrack", "ImprAssocTracker"],
    }
    candidates = _aliases.get(cfg.tracker_name, [cfg.tracker_name])
    namespaces = []
    try:
        from boxmot import trackers as bm_trackers
        namespaces.append(bm_trackers)
    except ImportError:
        pass
    namespaces.append(bm)
    cls = None
    for ns in namespaces:
        for name in candidates:
            cand = getattr(ns, name, None)
            if cand is not None:
                cls = cand
                break
        if cls is not None:
            break
    if cls is None:
        avail = sorted({n for ns in namespaces for n in dir(ns)
                        if n[:1].isupper()})
        raise ValueError(
            f"unknown BoxMOT tracker '{cfg.tracker_name}' "
            f"(tried aliases {candidates}); available={avail}"
        )

    # Translate our generic device string into torch.device for boxmot.
    # (Boxmot uses torch.device throughout; "mps" maps cleanly on Mac.)
    if cfg.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        device = torch.device(cfg.device if ":" in str(cfg.device) else "cuda")
    else:
        device = torch.device("cpu")

    # boxmot >=10.0.52 renamed the appearance-tracker constructor kwargs:
    #   reid_weights -> model_weights
    #   half         -> fp16
    # but kept HybridSORT on the legacy (reid_weights, half) signature
    # **and** added a required `det_thresh` positional. We introspect the
    # __init__ signature so the same script keeps working across boxmot
    # versions without per-tracker special cases.
    import inspect
    common = dict(**cfg.extra_kwargs)
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if cfg.reid_weights is not None and (
        "model_weights" in params or "reid_weights" in params
    ):
        kw = dict(common)
        if "model_weights" in params:
            kw["model_weights"] = Path(cfg.reid_weights)
        elif "reid_weights" in params:
            kw["reid_weights"] = Path(cfg.reid_weights)
        if "device" in params:
            kw["device"] = device
        if "fp16" in params:
            kw["fp16"] = bool(cfg.half)
        elif "half" in params:
            kw["half"] = bool(cfg.half)
        # HybridSORT requires det_thresh.
        if "det_thresh" in params and "det_thresh" not in kw:
            default = params["det_thresh"].default
            kw["det_thresh"] = (default if default is not inspect.Parameter.empty
                                else 0.3)
        return cls(**kw)
    # ReID-free trackers
    return cls(**common)


# --------------------------------------------------------------------------
# Per-clip runner
# --------------------------------------------------------------------------

def _open_video(path: Path):
    """Open a video file and yield (frame_idx, frame_bgr) until EOF.

    Falls back from cv2 to imageio if cv2 is missing *or* fails to open
    the file (the NGC PyTorch image ships OpenCV without ffmpeg, so
    .mov clips open with cv2.isOpened() == False).
    On a Mac this should always be cv2.
    """
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
            cap.release()
            return
        cap.release()
        log.warning("cv2 could not open %s; falling back to imageio/FFMPEG", path)

    import imageio.v3 as iio
    for idx, frame in enumerate(iio.imiter(str(path), plugin="FFMPEG")):
        # imageio gives RGB; convert to BGR to match cv2 semantics.
        frame_bgr = frame[..., ::-1]
        yield idx, frame_bgr


def run_one_clip(
    *, clip: ClipSpec, cfg: BoxmotConfig, weights: Path,
    cache_root: Path, force: bool = False,
) -> Path:
    """Run YOLO + BoxMOT tracker on one clip and write the cache pkl."""
    cache_dir = cache_root / clip.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The filename starts with "imgsz...conf..." so the glob in
    # ``eval/overfit/_common.py::_list_cached_pkls`` finds it. With the
    # multi-scale ensemble we still report the *primary* imgsz in the
    # filename — the per-scale list is folded into ``cfg.hash()`` so
    # different ensembles never collide.
    primary_imgsz = (cfg.imgsz_ensemble[0]
                     if cfg.imgsz_ensemble else cfg.imgsz)
    cache_path = cache_dir / (
        f"imgsz{primary_imgsz}_conf{cfg.conf:.3f}_iou{cfg.iou:.3f}"
        f"_boxmot_{cfg.tracker_name.lower()}_{cfg.hash()}_max"
        f"{clip.max_frames if clip.max_frames is not None else 'all'}.pkl"
    )

    if cache_path.is_file() and not force:
        log.info("[%s] cache hit (%s)", clip.name, cache_path.name)
        return cache_path

    detect = _make_yolo_detector(
        weights, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, device=cfg.device,
        imgsz_ensemble=(list(cfg.imgsz_ensemble) if cfg.imgsz_ensemble else None),
        ensemble_iou=cfg.ensemble_iou,
    )
    tracker = _make_boxmot_tracker(cfg)

    out_frames: List[FrameDetections] = []
    t0 = time.time()
    n_processed = 0
    for idx, frame_bgr in _open_video(clip.video):
        if clip.max_frames is not None and idx >= clip.max_frames:
            break
        dets = detect(frame_bgr)             # (N, 6)
        # BoxMOT trackers expect a 2D array even for "no detections"
        # frames; pass an empty (0, 6) array if needed.
        if dets.size == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        # tracker.update returns (M, K) where K depends on the tracker
        # but the *last* numeric column is always the track id and the
        # first 4 columns are xyxy. The conf column is at index 5 for
        # every BoxMOT tracker we care about (DeepOcSort, BotSort,
        # OcSort, ByteTrack); to stay tracker-agnostic we read xyxy and
        # id by slot and pull conf from the conf column.
        try:
            tracks = tracker.update(dets, frame_bgr)
        except Exception as exc:
            log.exception("[%s] tracker.update failed at frame %d: %s",
                          clip.name, idx, exc)
            tracks = np.zeros((0, 7), dtype=np.float32)

        if tracks is None or len(tracks) == 0:
            out_frames.append(FrameDetections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            ))
        else:
            tracks = np.asarray(tracks, dtype=np.float32)
            xyxys = tracks[:, 0:4].astype(np.float32)
            # id column is conventionally 4 in BoxMOT (post-update layout
            # is [x1, y1, x2, y2, id, conf, cls, det_index]).
            tids = tracks[:, 4].astype(np.float32)
            confs = (
                tracks[:, 5].astype(np.float32)
                if tracks.shape[1] > 5
                else np.ones(len(tracks), dtype=np.float32)
            )
            out_frames.append(FrameDetections(xyxys, confs, tids))
        n_processed += 1

    dt = time.time() - t0
    log.info("[%s]   -> %d frames in %.1fs (%.1f fps)",
             clip.name, n_processed, dt, n_processed / max(dt, 1e-6))

    with open(cache_path, "wb") as f:
        pickle.dump(out_frames, f)
    log.info("[%s] wrote %s", clip.name, cache_path)
    return cache_path


# --------------------------------------------------------------------------
# Multi-clip driver
# --------------------------------------------------------------------------

def run_all_clips(
    *, cfg: BoxmotConfig, weights: Path, output_dir: Path,
    user_clips_root: Path, clip_filter: Optional[List[str]] = None,
    force: bool = False,
    extra_clips: Optional[List["ClipSpec"]] = None,
) -> Dict[str, Path]:
    cache_root = output_dir / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    clips = _all_clip_specs(user_clips_root)
    if extra_clips:
        clips = list(clips) + list(extra_clips)
    if clip_filter:
        wanted = set(clip_filter)
        clips = [c for c in clips if c.name in wanted]
    for clip in clips:
        if not clip.video.is_file():
            log.warning("[%s] video not found at %s -- skipping",
                        clip.name, clip.video)
            continue
        try:
            paths[clip.name] = run_one_clip(
                clip=clip, cfg=cfg, weights=weights,
                cache_root=cache_root, force=force,
            )
        except Exception as exc:
            log.exception("[%s] run failed: %s", clip.name, exc)
    return paths


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker", required=True,
                   help="BoxMOT tracker class name "
                        "(DeepOcSort | BotSort | OcSort | ByteTrack | "
                        "StrongSort | BoostTrack | HybridSort | SFSORT)")
    p.add_argument("--reid-weights", type=Path, default=None,
                   help="ReID model weights (auto-downloaded by BoxMOT if "
                        "you supply a known model name like "
                        "osnet_x0_25_msmt17.pt). Required for DeepOcSort, "
                        "BotSort+ReID, StrongSort, BoostTrack, HybridSort. "
                        "Defaults to osnet_x0_25_msmt17.pt for those.")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--imgsz-ensemble", type=int, nargs="+", default=None,
                   help="Phase 2 multi-scale detection ensemble. Pass two or "
                        "more imgsz values; YOLO runs at each scale and "
                        "per-frame boxes are NMS-unioned at --ensemble-iou. "
                        "Overrides --imgsz when set. Example: "
                        "`--imgsz-ensemble 768 1024`. ")
    p.add_argument("--ensemble-iou", type=float, default=0.6,
                   help="NMS IoU threshold for the cross-scale union. "
                        "Default 0.6 — tight enough to fuse same-person "
                        "duplicates, loose enough to keep close neighbours.")
    p.add_argument("--conf", type=float, default=0.31)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--device", default="mps")
    p.add_argument("--half", action="store_true",
                   help="ReID half-precision (fp16). Off by default.")
    p.add_argument("--clips", nargs="*", default=None,
                   help="Restrict to a subset of clip names.")
    p.add_argument("--extra-clip", action="append", default=[],
                   metavar="NAME=PATH",
                   help="Register an ad-hoc clip (e.g. unseen footage) by "
                        "passing NAME=ABSOLUTE_PATH; can be repeated. The "
                        "clip is appended to the registry and is then "
                        "selectable via --clips.")
    p.add_argument("--weights", type=Path,
                   default=REPO_ROOT / "weights" / "best.pt")
    p.add_argument("--user-clips-root", type=Path,
                   default=DEFAULT_USER_CLIPS_ROOT)
    p.add_argument("--output", type=Path, required=True,
                   help="Output dir. Cache lands at <output>/_cache/<clip>/*.pkl")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if a cache file already exists.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _default_reid_for(tracker_name: str) -> Optional[Path]:
    """Pick a sensible default ReID checkpoint for trackers that need one.

    BoxMOT's ReID loader auto-downloads anything in its registry if the
    file isn't on disk, so we just supply the canonical small OSNet
    checkpoint.
    """
    needs_reid = {
        "DeepOcSort", "BotSort", "StrongSort", "BoostTrack", "HybridSort",
    }
    if tracker_name not in needs_reid:
        return None
    # OSNet x0.25 trained on MSMT17 is the standard small/fast default.
    return Path("osnet_x0_25_msmt17.pt")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    reid_weights = args.reid_weights
    if reid_weights is None:
        reid_weights = _default_reid_for(args.tracker)
        if reid_weights is not None:
            log.info("[%s] using default ReID weights: %s",
                     args.tracker, reid_weights)

    cfg = BoxmotConfig(
        tracker_name=args.tracker,
        reid_weights=reid_weights,
        imgsz=int(args.imgsz),
        imgsz_ensemble=(tuple(sorted(set(int(s) for s in args.imgsz_ensemble)))
                        if args.imgsz_ensemble else None),
        ensemble_iou=float(args.ensemble_iou),
        conf=float(args.conf),
        iou=float(args.iou),
        device=str(args.device),
        half=bool(args.half),
    )
    log.info("config: %s", cfg)
    extra_clips: List[ClipSpec] = []
    for raw in args.extra_clip:
        if "=" not in raw:
            raise ValueError(f"--extra-clip expected NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        extra_clips.append(ClipSpec(name=name.strip(),
                                    video=Path(path).expanduser().resolve(),
                                    leaked=False))
    paths = run_all_clips(
        cfg=cfg, weights=args.weights, output_dir=args.output,
        user_clips_root=args.user_clips_root, clip_filter=args.clips,
        force=bool(args.force), extra_clips=extra_clips,
    )
    log.info("done. %d clip caches at %s/_cache/", len(paths), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
