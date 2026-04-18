"""Shared MOT scoring helpers for both the local and remote sweeps.

Single entry point: ``score_tracks_dict(tracks, num_frames, gt_path)`` returns
a dict of CLEAR / ID metrics (from py-motmetrics, IoU 0.5) plus an optional
HOTA value (from TrackEval if installed) plus the composite ``score`` we
rank by.

This is the same composite ranking used by ``eval/compare_trackers.py``
(line 225): ``0.5 * IDF1 + 0.5 * HOTA`` with HOTA falling back to IDF1
when TrackEval is missing.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mot_eval_utils import mot_metrics_for_sequence, frame_detections_to_mot_rows
from prune_tracks import FrameDetections
from tracking.postprocess import Track, tracks_to_frame_detections


log = logging.getLogger(__name__)


def _try_hota(gt_path: Path, pred_frames: List[FrameDetections], tmp_root: Path) -> Optional[float]:
    """Compute HOTA using TrackEval if importable; else return None.

    TrackEval expects MOT-Challenge folder layout. We materialise the
    minimum required directory structure inside ``tmp_root`` and then
    invoke ``HOTA.compute`` on a single sequence.
    """
    try:
        from trackeval.metrics import HOTA  # type: ignore
        from trackeval.datasets import MotChallenge2DBox  # type: ignore
        import trackeval  # type: ignore
    except Exception:
        return None
    try:
        # MOTChallenge layout:
        #   <tmp>/gt/<benchmark>/<seq>/gt/gt.txt
        #   <tmp>/gt/<benchmark>/<seq>/seqinfo.ini
        #   <tmp>/trackers/<benchmark>/<tracker>/data/<seq>.txt
        bench = "SWEEP"
        seq = "seq01"
        tracker_name = "candidate"
        gt_dir = tmp_root / "gt" / bench / seq
        (gt_dir / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy(str(gt_path), str(gt_dir / "gt" / "gt.txt"))
        # seqinfo.ini
        n_frames = max((sum(1 for _ in pred_frames),), default=0)
        seqinfo = (
            "[Sequence]\n"
            f"name={seq}\n"
            f"seqLength={n_frames}\n"
            "frameRate=30\n"
            "imWidth=1920\nimHeight=1080\n"
        )
        (gt_dir / "seqinfo.ini").write_text(seqinfo)

        track_dir = tmp_root / "trackers" / bench / tracker_name / "data"
        track_dir.mkdir(parents=True, exist_ok=True)
        rows = frame_detections_to_mot_rows(pred_frames)
        (track_dir / f"{seq}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))

        seqmaps = tmp_root / "gt" / bench / "seqmaps"
        seqmaps.mkdir(parents=True, exist_ok=True)
        (seqmaps / f"{bench}-train.txt").write_text(f"name\n{seq}\n")

        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False,
                         "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False,
                         "PLOT_CURVES": False, "USE_PARALLEL": False})
        ds_cfg = MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER": str(tmp_root / "gt"),
            "TRACKERS_FOLDER": str(tmp_root / "trackers"),
            "BENCHMARK": bench,
            "SPLIT_TO_EVAL": "train",
            "TRACKERS_TO_EVAL": [tracker_name],
            "SEQ_INFO": {seq: n_frames},
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
        })
        evaluator = trackeval.Evaluator(eval_cfg)
        ds = MotChallenge2DBox(ds_cfg)
        out, _ = evaluator.evaluate([ds], [HOTA()])
        # navigate the deeply nested return dict
        try:
            seq_res = out["MotChallenge2DBox"][tracker_name][seq]
            hota_arr = seq_res["pedestrian"]["HOTA"]["HOTA"]
            return float(np.mean(hota_arr))
        except Exception:
            return None
    except Exception as exc:
        log.debug("TrackEval HOTA failed: %s", exc)
        return None


def score_tracks_dict(
    tracks: Dict[int, Track],
    num_frames: int,
    gt_path: Path,
    *,
    enable_hota: bool = False,
) -> Dict[str, float]:
    """Score a postprocessed dict of tracks against MOT GT.

    Returns CLEAR + ID metrics plus ``hota`` (NaN if not computed) and
    ``score`` (composite ranking).
    """
    frames = tracks_to_frame_detections(tracks, num_frames=num_frames)
    return score_frame_detections(frames, gt_path, enable_hota=enable_hota)


def score_frame_detections(
    pred_frames: List[FrameDetections],
    gt_path: Path,
    *,
    enable_hota: bool = False,
) -> Dict[str, float]:
    """Score a list of FrameDetections directly. Used by the legacy path."""
    metrics = mot_metrics_for_sequence(gt_path, pred_frames)

    hota = float("nan")
    if enable_hota:
        with tempfile.TemporaryDirectory(prefix="trackeval_") as td:
            v = _try_hota(gt_path, pred_frames, Path(td))
            if v is not None:
                hota = float(v)
    metrics["hota"] = hota

    idf1 = float(metrics.get("idf1", 0.0))
    if not np.isfinite(idf1):
        idf1 = 0.0
    h_or_idf = hota if np.isfinite(hota) else idf1
    metrics["score"] = 0.5 * idf1 + 0.5 * h_or_idf
    return metrics
