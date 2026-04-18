"""Per-clip comparison metrics for the dual 3D pipeline (plan Task 11 + Followup #1).

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

from typing import Optional, Tuple

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


# ---------------------------------------------------------------------------
# Procrustes alignment + PA-MPJPE (Followup #1)
# ---------------------------------------------------------------------------


def _procrustes_fit(
    a: np.ndarray,
    b: np.ndarray,
    allow_scale: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Estimate (R, t, s) such that ``s * b @ R.T + t`` ≈ ``a`` over valid points.

    Implements the classic Kabsch algorithm with Umeyama's uniform-scale
    extension. Frames where either input has any NaN are excluded from the
    fit. Returns ``(None, None, 1.0)`` if fewer than 3 valid points remain
    (no unique rigid solution).
    """
    a_flat = a.reshape(-1, 3)
    b_flat = b.reshape(-1, 3)
    valid = ~(np.isnan(a_flat).any(axis=1) | np.isnan(b_flat).any(axis=1))
    if int(valid.sum()) < 3:
        return None, None, 1.0
    A = a_flat[valid].astype(np.float64, copy=False)
    B = b_flat[valid].astype(np.float64, copy=False)
    a_c = A.mean(axis=0)
    b_c = B.mean(axis=0)
    Ac = A - a_c
    Bc = B - b_c
    H = Bc.T @ Ac
    U, S, Vt = np.linalg.svd(H)
    d_sign = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d_sign == 0.0:
        d_sign = 1.0
    D = np.diag([1.0, 1.0, d_sign])
    R = Vt.T @ D @ U.T
    if allow_scale:
        var_b = float((Bc * Bc).sum())
        if var_b > 0.0:
            s = float((S * np.array([1.0, 1.0, d_sign])).sum() / var_b)
        else:
            s = 1.0
    else:
        s = 1.0
    t = a_c - s * (R @ b_c)
    return R, t, s


def _apply_transform(
    b: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """Apply ``b' = s * b @ R.T + t`` to a (..., 3) tensor, preserving NaN rows."""
    flat = b.reshape(-1, 3).astype(np.float64, copy=False)
    nan_mask = np.isnan(flat).any(axis=1)
    transformed = s * (flat @ R.T) + t
    transformed[nan_mask] = np.nan
    return transformed.reshape(b.shape)


def align_procrustes(
    a: np.ndarray,
    b: np.ndarray,
    *,
    per_dancer: bool = True,
    allow_scale: bool = False,
) -> np.ndarray:
    """Align ``b`` to ``a`` with the optimal rigid (or rigid+scale) transform.

    Parameters
    ----------
    a : np.ndarray
        Reference joints, ``(T, N_dancers, J, 3)``.
    b : np.ndarray
        Joints to align to ``a``. Must have the same shape.
    per_dancer : bool, default True
        Estimate one ``(R, t)`` per dancer (recommended for cross-pipeline
        comparison since each dancer's root translation is independent).
        Set to ``False`` to compute one global transform across all dancers.
    allow_scale : bool, default False
        Estimate a uniform scale ``s`` (Umeyama). Useful when comparing
        pipelines that produce meshes at different metric scales.

    Returns
    -------
    np.ndarray
        ``b`` after alignment, same shape as ``b``, dtype float64. Frames
        where ``a`` or ``b`` had any NaN are excluded from the fit and
        emitted as NaN in the output. Dancers with fewer than 3 valid
        ``(frame, joint)`` points are returned untransformed (still NaN
        if they were all-NaN to start).
    """
    if a.shape != b.shape:
        raise ValueError(
            f"align_procrustes requires identical shapes, got {a.shape} vs {b.shape}"
        )
    out = b.astype(np.float64, copy=True)
    if per_dancer:
        for d in range(a.shape[1]):
            R, t, s = _procrustes_fit(a[:, d], b[:, d], allow_scale)
            if R is None:
                continue
            out[:, d] = _apply_transform(b[:, d], R, t, s)
        return out
    R, t, s = _procrustes_fit(a, b, allow_scale)
    if R is None:
        return out
    return _apply_transform(b, R, t, s)


def per_joint_mpjpe_pa(
    a: np.ndarray,
    b: np.ndarray,
    *,
    per_dancer: bool = True,
    allow_scale: bool = False,
) -> np.ndarray:
    """MPJPE after Procrustes alignment of ``b`` to ``a``.

    Mirrors :func:`per_joint_mpjpe` but applies :func:`align_procrustes`
    first so the metric reflects per-joint pose error rather than
    coordinate-frame mismatch. Same shape and NaN semantics as
    :func:`per_joint_mpjpe`.
    """
    b_aligned = align_procrustes(
        a, b, per_dancer=per_dancer, allow_scale=allow_scale
    )
    return per_joint_mpjpe(a, b_aligned)
