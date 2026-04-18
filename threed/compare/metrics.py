"""Per-clip comparison metrics for the dual 3D pipeline (plan Task 11).

All metrics operate on a stacked joints tensor of shape::

    joints[T, N_dancers, J, 3]

Conventions:

- ``T``  is the frame count (the *intersection* of frames produced by
  the two pipelines — Stage D pads missing frames with NaN before
  passing arrays in here).
- ``N_dancers`` is the number of matched DeepOcSort track IDs (since
  both pipelines reuse our shared track IDs, dancer ``d`` in pipeline A
  IS dancer ``d`` in pipeline B without any identity assignment step).
- ``J`` is the number of joints in the common skeleton (we use COCO-17
  via :mod:`threed.compare.joints`, but these helpers are
  skeleton-agnostic).
- The last axis is ``(x, y, z)`` in metres.

NaN frames (a dancer not detected on a particular frame) are ignored
via ``np.nanmean`` so a partially-tracked dancer doesn't bias the
metric. Use ``np.isnan(out)`` downstream to spot dancers that were
NEVER detected by a given pipeline.
"""
from __future__ import annotations

import numpy as np


def per_joint_jitter(joints: np.ndarray) -> np.ndarray:
    """Mean inter-frame velocity per (dancer, joint) in metres / frame.

    Computed as the L2 norm of the temporal first difference, then
    averaged over time with ``np.nanmean``. NaN frames (dancer not
    detected) are excluded.

    Parameters
    ----------
    joints : np.ndarray
        ``(T, N_dancers, J, 3)`` array.

    Returns
    -------
    np.ndarray
        ``(N_dancers, J)`` array of per-joint mean velocities.
        NaN if a dancer has zero detected pairs.

    Notes
    -----
    A noiseless static joint returns 0; a joint moving at a constant
    ``v`` m/frame returns exactly ``v``. This matches the upstream PHALP
    "jitter" metric in
    https://github.com/brjathu/PHALP/blob/main/PHALP/utils/metrics.py.
    Used by Stage D as a *per-pipeline* smoothness diagnostic — if
    PromptHMR jitter is much greater than SAM-Body4D for the same
    track, the SMPL-X fit is noisier (likely needs ``run_post_opt``).
    """
    diffs = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(diffs, axis=0)


def per_joint_mpjpe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean per-joint position error between two joint sequences.

    Uses ``np.nanmean`` so frames with NaN in either array are
    excluded. Both arrays MUST be the same shape.

    Parameters
    ----------
    a, b : np.ndarray
        Two ``(T, N_dancers, J, 3)`` arrays in the SAME coordinate
        frame (cam-coords for our Stage D — see
        ``threed/compare/joints_extract.py``).

    Returns
    -------
    np.ndarray
        ``(N_dancers, J)`` array of mean per-joint position errors in
        metres. NaN if a dancer has no detected frames in either
        pipeline.

    Notes
    -----
    No Procrustes alignment is applied — this is the *raw* MPJPE in
    the cam frame (Stage D-extract aligns both pipelines to the cam
    frame so the comparison is fair). For a frame-invariant metric,
    use jitter / foot-skating instead, or layer Procrustes on top
    later.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"per_joint_mpjpe requires identical shapes, got {a.shape} vs {b.shape}"
        )
    diff = np.linalg.norm(a - b, axis=-1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(diff, axis=0)


def foot_skating(
    joints: np.ndarray,
    *,
    foot_idx: int = 15,
    threshold: float = 0.05,
) -> np.ndarray:
    """Mean foot velocity for "planted" frames (foot height < threshold).

    A planted foot that nonetheless moves laterally is "skating" — the
    classic 3D HMR failure mode. We average the L2 velocity over all
    frames where the foot is below ``threshold`` metres above the
    ground plane (``z`` axis, +z is up). NaN frames are ignored.

    Parameters
    ----------
    joints : np.ndarray
        ``(T, N_dancers, J, 3)`` array. ``z`` is up.
    foot_idx : int, default 15
        Index of the foot joint in the skeleton — defaults to COCO-17
        ``left_ankle``. Pass ``foot_idx=16`` for ``right_ankle``.
    threshold : float, default 0.05
        Maximum height (m) for the foot to be considered "planted".

    Returns
    -------
    np.ndarray
        ``(N_dancers,)`` array of mean planted-foot velocities in m/frame.
        ``0.0`` if a dancer has no planted frames (that's "no
        evidence", not "no skating" — interpret with care).

    Notes
    -----
    For a static camera with a known ground plane, COCO-17 ankles in
    cam-coords have ``z`` proportional to *depth* (forward-from-camera),
    not vertical height. This matters: for tripod shots the "foot
    height" is actually a forward distance and the threshold should be
    much larger. Stage D documents this caveat in its README; for our
    canonical clips the camera is roughly chest-high pointing slightly
    down, so an ankle at z<0.05m below pelvis is a reasonable
    "planted" heuristic. Reinterpret per-clip if needed.
    """
    foot = joints[:, :, foot_idx, :]
    h = foot[:, :, 2]
    planted = h < threshold
    vel = np.linalg.norm(np.diff(foot, axis=0), axis=-1)
    mask = planted[1:] & ~np.isnan(vel)
    out = np.zeros(joints.shape[1], dtype=np.float32)
    for d in range(joints.shape[1]):
        m = mask[:, d]
        if m.any():
            out[d] = float(vel[m, d].mean())
    return out
