"""Scorer for the `2pplTest` clip — 'exactly two people throughout'.

The 2pplTest video has no MOT GT. Instead we score a tracker output
against the structural prior that the video contains exactly two
people for its entire duration.

A perfect score (1.0) requires:
  - Every frame contains exactly 2 distinct active track ids.
  - The whole video contains exactly 2 distinct track ids ever.
  - Each of the 2 tracks persists for at least `persistence` of the
    video (default 80%).

Anything less than perfect gets a graded penalty so Optuna's TPE
sampler can climb a smooth gradient toward 1.0:

  score = frac_exactly_two
        * min(1.0, 2.0 / max(unique_ids, 1))
        * (1.0 if persistence_ok else 0.5)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from tracking.postprocess import Track


def score_2ppl(
    tracks: Mapping[int, Track],
    num_frames: int,
    *,
    persistence: float = 0.80,
) -> Dict[str, float]:
    """Score a postprocessed `dict[track_id -> Track]` against the
    'exactly two people throughout' prior. Returns a dict with the
    composite ``score`` plus diagnostic fields."""

    if num_frames <= 0:
        return {
            "score": 0.0,
            "frac_exactly_two": 0.0,
            "unique_ids": 0,
            "persistence_ok": False,
            "min_coverage": 0.0,
            "num_frames": 0,
        }

    per_frame_count = np.zeros(num_frames, dtype=np.int32)
    coverages: Dict[int, int] = {}
    for tid, t in tracks.items():
        # Track.frames lists every frame this track is active in (post-interp).
        frames = np.asarray(t.frames, dtype=np.int64)
        frames = frames[(frames >= 0) & (frames < num_frames)]
        if frames.size == 0:
            continue
        per_frame_count[frames] += 1
        coverages[int(tid)] = int(frames.size)

    unique_ids = len(coverages)
    frac_exactly_two = float(np.mean(per_frame_count == 2))

    if unique_ids == 0:
        min_coverage = 0.0
        persistence_ok = False
    else:
        min_coverage = float(min(coverages.values()) / num_frames)
        persistence_ok = all(
            (cov / num_frames) >= persistence for cov in coverages.values()
        )

    if frac_exactly_two == 1.0 and unique_ids == 2 and persistence_ok:
        score = 1.0
    else:
        id_factor = min(1.0, 2.0 / max(unique_ids, 1))
        persist_factor = 1.0 if persistence_ok else 0.5
        score = float(frac_exactly_two * id_factor * persist_factor)

    return {
        "score": float(score),
        "frac_exactly_two": float(frac_exactly_two),
        "unique_ids": int(unique_ids),
        "persistence_ok": bool(persistence_ok),
        "min_coverage": float(min_coverage),
        "num_frames": int(num_frames),
    }
