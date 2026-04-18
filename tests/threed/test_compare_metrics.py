"""Unit tests for :mod:`threed.compare.metrics` (plan Task 11b + Followup #1)."""
from __future__ import annotations

import numpy as np
import pytest

from threed.compare.metrics import (
    align_procrustes,
    foot_skating,
    foot_skating_world_frame,
    mpjpe_2d,
    per_joint_jitter,
    per_joint_mpjpe,
    per_joint_mpjpe_pa,
    reproject_3d_to_2d,
)


# ---------------------------------------------------------------------------
# per_joint_jitter
# ---------------------------------------------------------------------------


class TestPerJointJitter:
    def test_returns_zero_for_constant(self):
        j = np.zeros((10, 1, 17, 3), dtype=np.float32)
        out = per_joint_jitter(j)
        assert out.shape == (1, 17)
        np.testing.assert_array_equal(out, np.zeros((1, 17)))

    def test_picks_up_constant_motion(self):
        """A joint moving 0.1 m / frame should jitter at exactly 0.1 m / frame."""
        j = np.zeros((10, 1, 17, 3), dtype=np.float32)
        j[:, 0, 0, 0] = np.arange(10) * 0.1
        out = per_joint_jitter(j)
        np.testing.assert_allclose(out[0, 0], 0.1, rtol=1e-5)
        for k in range(1, 17):
            assert out[0, k] == 0.0

    def test_handles_nan_frames(self):
        """NaN frames must be ignored (a dancer not detected on some frames)."""
        j = np.zeros((10, 2, 17, 3), dtype=np.float32)
        j[:, 1] = np.nan
        out = per_joint_jitter(j)
        assert out.shape == (2, 17)
        np.testing.assert_array_equal(out[0], np.zeros(17))
        assert np.isnan(out[1]).all()

    def test_returns_per_dancer_per_joint(self):
        j = np.zeros((5, 3, 17, 3), dtype=np.float32)
        j[:, 0, 5, 0] = np.arange(5) * 0.2
        j[:, 1, 9, 1] = np.arange(5) * 0.5
        out = per_joint_jitter(j)
        assert out.shape == (3, 17)
        np.testing.assert_allclose(out[0, 5], 0.2, rtol=1e-5)
        np.testing.assert_allclose(out[1, 9], 0.5, rtol=1e-5)
        assert out[2].sum() == 0.0


# ---------------------------------------------------------------------------
# per_joint_mpjpe
# ---------------------------------------------------------------------------


class TestPerJointMpjpe:
    def test_zero_when_identical(self):
        rng = np.random.default_rng(0)
        a = rng.random((10, 1, 17, 3)).astype(np.float32)
        out = per_joint_mpjpe(a, a.copy())
        np.testing.assert_array_equal(out, np.zeros((1, 17)))

    def test_constant_offset(self):
        a = np.zeros((10, 1, 17, 3), dtype=np.float32)
        b = a.copy()
        b[:, :, :, 0] = 0.5
        out = per_joint_mpjpe(a, b)
        np.testing.assert_allclose(out, np.full((1, 17), 0.5), rtol=1e-6)

    def test_handles_nan_frames(self):
        a = np.zeros((10, 1, 17, 3), dtype=np.float32)
        b = a.copy()
        b[:, :, :, 0] = 1.0
        b[3] = np.nan
        out = per_joint_mpjpe(a, b)
        np.testing.assert_allclose(out, np.full((1, 17), 1.0), rtol=1e-6)

    def test_shape_mismatch_raises(self):
        a = np.zeros((10, 1, 17, 3), dtype=np.float32)
        b = np.zeros((10, 2, 17, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            per_joint_mpjpe(a, b)


# ---------------------------------------------------------------------------
# foot_skating
# ---------------------------------------------------------------------------


class TestFootSkating:
    def test_planted_foot_zero_velocity(self):
        """Foot constant under threshold -> skate=0 (planted but not moving)."""
        j = np.zeros((10, 1, 17, 3), dtype=np.float32)
        out = foot_skating(j, foot_idx=15, threshold=0.05)
        assert out.shape == (1,)
        assert out[0] == 0.0

    def test_planted_foot_with_motion(self):
        """Foot below threshold + sliding -> skate equals horizontal velocity."""
        j = np.zeros((10, 1, 17, 3), dtype=np.float32)
        j[:, 0, 15, 0] = np.arange(10) * 0.05
        out = foot_skating(j, foot_idx=15, threshold=0.05)
        np.testing.assert_allclose(out[0], 0.05, rtol=1e-5)

    def test_above_threshold_ignored(self):
        """Foot lifted (above threshold) -> not counted, skate=0."""
        j = np.zeros((10, 1, 17, 3), dtype=np.float32)
        j[:, 0, 15, 2] = 1.0
        j[:, 0, 15, 0] = np.arange(10) * 0.5
        out = foot_skating(j, foot_idx=15, threshold=0.05)
        assert out[0] == 0.0

    def test_per_dancer_independent(self):
        """Two dancers, only one slides -> only that dancer's skate is non-zero."""
        j = np.zeros((10, 2, 17, 3), dtype=np.float32)
        j[:, 0, 15, 0] = np.arange(10) * 0.1
        out = foot_skating(j, foot_idx=15, threshold=0.05)
        np.testing.assert_allclose(out[0], 0.1, rtol=1e-5)
        assert out[1] == 0.0


# ---------------------------------------------------------------------------
# align_procrustes (Followup #1)
# ---------------------------------------------------------------------------


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Build a uniformly-random rotation matrix via QR of a Gaussian matrix.

    Tests below need an arbitrary 3-D rotation; this returns one that is
    a proper (det=+1) rotation so we can assert "transform is recovered
    by the aligner" rather than "transform OR its mirror is recovered".
    """
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class TestAlignProcrustes:
    def test_identity_returns_input_unchanged(self):
        """If b == a already, the aligner should be a no-op (within float eps)."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((10, 2, 17, 3)).astype(np.float64)
        b_aligned = align_procrustes(a, a.copy())
        np.testing.assert_allclose(b_aligned, a, atol=1e-10)

    def test_pure_translation_recovered(self):
        """b = a + t0 must reduce to a after alignment (PA-MPJPE -> 0)."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((20, 1, 17, 3)).astype(np.float64)
        t0 = np.array([1.5, -3.2, 7.1])
        b = a + t0
        b_aligned = align_procrustes(a, b)
        np.testing.assert_allclose(b_aligned, a, atol=1e-9)

    def test_pure_rotation_recovered(self):
        """b = R0 @ a must reduce to a after alignment (PA-MPJPE -> 0)."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((20, 1, 17, 3)).astype(np.float64)
        R0 = _random_rotation(rng)
        b = a @ R0.T
        b_aligned = align_procrustes(a, b)
        np.testing.assert_allclose(b_aligned, a, atol=1e-9)

    def test_per_dancer_independent_transforms(self):
        """Two dancers with DIFFERENT rotations recovered independently when per_dancer=True."""
        rng = np.random.default_rng(3)
        a = rng.standard_normal((20, 2, 17, 3)).astype(np.float64)
        R1 = _random_rotation(rng)
        R2 = _random_rotation(rng)
        b = a.copy()
        b[:, 0] = a[:, 0] @ R1.T + np.array([1.0, 2.0, 3.0])
        b[:, 1] = a[:, 1] @ R2.T + np.array([-4.0, 5.0, -6.0])
        b_aligned = align_procrustes(a, b, per_dancer=True)
        np.testing.assert_allclose(b_aligned, a, atol=1e-9)

    def test_per_dancer_false_uses_single_transform(self):
        """With per_dancer=False the aligner can't fit two dancers' different rotations."""
        rng = np.random.default_rng(4)
        a = rng.standard_normal((20, 2, 17, 3)).astype(np.float64)
        R1 = _random_rotation(rng)
        R2 = _random_rotation(rng)
        b = a.copy()
        b[:, 0] = a[:, 0] @ R1.T
        b[:, 1] = a[:, 1] @ R2.T
        b_aligned = align_procrustes(a, b, per_dancer=False)
        global_err = float(np.linalg.norm(a - b_aligned, axis=-1).mean())
        per_dancer = align_procrustes(a, b, per_dancer=True)
        per_err = float(np.linalg.norm(a - per_dancer, axis=-1).mean())
        assert per_err < 1e-6
        assert global_err > 0.05

    def test_uniform_scale_recovered_when_enabled(self):
        """With allow_scale=True, b = 2*a aligns back to a; without it doesn't."""
        rng = np.random.default_rng(5)
        a = rng.standard_normal((20, 1, 17, 3)).astype(np.float64)
        b = 2.0 * a
        no_scale = align_procrustes(a, b, allow_scale=False)
        with_scale = align_procrustes(a, b, allow_scale=True)
        no_scale_err = float(np.linalg.norm(a - no_scale, axis=-1).mean())
        with_scale_err = float(np.linalg.norm(a - with_scale, axis=-1).mean())
        assert with_scale_err < 1e-9
        assert no_scale_err > 0.1

    def test_nan_frames_excluded_from_fit_but_preserved_in_output(self):
        """NaN frames in b must not bias the (R,t) estimate, and must remain NaN in output."""
        rng = np.random.default_rng(6)
        a = rng.standard_normal((10, 1, 17, 3)).astype(np.float64)
        t0 = np.array([1.0, 2.0, 3.0])
        b = a + t0
        b[3] = np.nan
        b_aligned = align_procrustes(a, b)
        assert np.isnan(b_aligned[3]).all()
        for f in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
            np.testing.assert_allclose(b_aligned[f], a[f], atol=1e-9)

    def test_dancer_with_all_nan_returns_all_nan(self):
        """If a dancer has zero valid frames, the per-dancer fit can't run -> output stays NaN."""
        rng = np.random.default_rng(7)
        a = rng.standard_normal((10, 2, 17, 3)).astype(np.float64)
        b = a.copy()
        b[:, 1] = np.nan
        b_aligned = align_procrustes(a, b, per_dancer=True)
        np.testing.assert_allclose(b_aligned[:, 0], a[:, 0], atol=1e-10)
        assert np.isnan(b_aligned[:, 1]).all()

    def test_shape_mismatch_raises(self):
        a = np.zeros((10, 1, 17, 3), dtype=np.float64)
        b = np.zeros((10, 2, 17, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            align_procrustes(a, b)


# ---------------------------------------------------------------------------
# per_joint_mpjpe_pa (Followup #1)
# ---------------------------------------------------------------------------


class TestPerJointMpjpePa:
    def test_zero_when_only_translation_differs(self):
        """Pure-translation b -> PA-MPJPE ~ 0 (raw MPJPE would be |t0|)."""
        rng = np.random.default_rng(8)
        a = rng.standard_normal((20, 1, 17, 3)).astype(np.float64)
        b = a + np.array([3.0, 4.0, 0.0])
        pa = per_joint_mpjpe_pa(a, b)
        raw = per_joint_mpjpe(a, b)
        assert pa.shape == (1, 17)
        np.testing.assert_allclose(pa, np.zeros_like(pa), atol=1e-9)
        np.testing.assert_allclose(raw, np.full_like(raw, 5.0), rtol=1e-6)

    def test_pa_mean_le_raw_mean_on_random_data(self):
        """Procrustes minimizes total squared error -> aggregate-mean PA <= aggregate-mean raw.

        Per-joint values can rise after alignment because Procrustes balances the fit
        across joints; the OVERALL mean across (dancer, joint) is what's guaranteed
        to be optimal.
        """
        rng = np.random.default_rng(9)
        a = rng.standard_normal((30, 3, 17, 3)).astype(np.float64)
        b = rng.standard_normal((30, 3, 17, 3)).astype(np.float64)
        pa_mean = float(np.nanmean(per_joint_mpjpe_pa(a, b)))
        raw_mean = float(np.nanmean(per_joint_mpjpe(a, b)))
        assert pa_mean <= raw_mean + 1e-9

    def test_handles_nan_frames(self):
        """NaN frames must not break the metric or the alignment."""
        rng = np.random.default_rng(10)
        a = rng.standard_normal((10, 1, 17, 3)).astype(np.float64)
        b = a + np.array([1.0, 2.0, 3.0])
        b[5] = np.nan
        pa = per_joint_mpjpe_pa(a, b)
        np.testing.assert_allclose(pa, np.zeros_like(pa), atol=1e-9)


# ---------------------------------------------------------------------------
# foot_skating_world_frame (Followup #2)
# ---------------------------------------------------------------------------


class TestFootSkatingWorldFrame:
    """World-frame foot-skating uses PHMR's ``joints_world.npy`` (SMPL-22 with
    Y-up). Unlike the cam-frame ``foot_skating``, here the ground plane is
    *calibrated per-dancer* by taking each dancer's lowest foot height as a
    proxy for the floor. This means the metric is robust to:

    1. tripod cameras above floor level (cam-frame Y has no zero crossing),
    2. dancers of different heights,
    3. clips on slightly slanted floors.

    The metric returns a per-dancer mean velocity (m / frame) of "planted"
    foot positions, where "planted" is ``height_above_floor < threshold_m``.
    """

    def test_planted_foot_zero_when_static(self):
        """Foot at floor height, no motion -> skate=0."""
        j = np.zeros((10, 1, 22, 3), dtype=np.float64)
        out = foot_skating_world_frame(j)
        assert out.shape == (1,)
        assert out[0] == 0.0

    def test_skating_picks_up_horizontal_motion(self):
        """Foot at floor + sliding 0.1 m/frame in X -> skate ~ 0.1."""
        j = np.zeros((10, 1, 22, 3), dtype=np.float64)
        j[:, 0, 7, 0] = np.arange(10) * 0.1
        out = foot_skating_world_frame(j, foot_idx=7)
        np.testing.assert_allclose(out[0], 0.1, rtol=1e-6)

    def test_lifted_foot_not_counted(self):
        """Foot rising above per-dancer floor -> lifted frames not counted."""
        j = np.zeros((10, 1, 22, 3), dtype=np.float64)
        j[0, 0, 7, 1] = 0.0
        j[1:, 0, 7, 1] = 1.0
        j[:, 0, 7, 0] = np.arange(10) * 0.5
        out = foot_skating_world_frame(j, foot_idx=7, threshold_m=0.05)
        assert out[0] == 0.0

    def test_per_dancer_floor_calibration(self):
        """A dancer entirely above Y=10 should still detect their planted foot."""
        j = np.zeros((10, 1, 22, 3), dtype=np.float64)
        j[:, 0, 7, 1] = 10.0
        j[:, 0, 7, 0] = np.arange(10) * 0.05
        out = foot_skating_world_frame(j, foot_idx=7, threshold_m=0.05)
        np.testing.assert_allclose(out[0], 0.05, rtol=1e-6)

    def test_per_dancer_independent(self):
        """Two dancers, only one slides -> only that dancer's skate is non-zero."""
        j = np.zeros((10, 2, 22, 3), dtype=np.float64)
        j[:, 0, 7, 0] = np.arange(10) * 0.2
        out = foot_skating_world_frame(j, foot_idx=7)
        np.testing.assert_allclose(out[0], 0.2, rtol=1e-6)
        assert out[1] == 0.0

    def test_handles_nan_frames(self):
        """NaN frames must be excluded from velocity computation."""
        j = np.zeros((10, 1, 22, 3), dtype=np.float64)
        j[:, 0, 7, 0] = np.arange(10) * 0.1
        j[5, 0, 7] = np.nan
        out = foot_skating_world_frame(j, foot_idx=7)
        assert np.isfinite(out[0])
        assert out[0] > 0.0

    def test_dancer_with_all_nan_returns_zero(self):
        """Dancer entirely undetected -> 0.0 (matches cam-frame foot_skating contract)."""
        j = np.zeros((10, 2, 22, 3), dtype=np.float64)
        j[:, 1] = np.nan
        out = foot_skating_world_frame(j, foot_idx=7)
        assert out[0] == 0.0
        assert out[1] == 0.0

    def test_uses_y_axis_as_height(self):
        """Default ``height_axis=1`` (Y is up) — sanity check the docs."""
        j = np.zeros((10, 1, 22, 3), dtype=np.float64)
        j[:, 0, 7, 2] = 1.0
        j[:, 0, 7, 0] = np.arange(10) * 0.1
        out = foot_skating_world_frame(j, foot_idx=7, threshold_m=0.05)
        np.testing.assert_allclose(out[0], 0.1, rtol=1e-6)


# ---------------------------------------------------------------------------
# reproject_3d_to_2d + mpjpe_2d (Followup #4)
# ---------------------------------------------------------------------------


class TestReproject3dTo2d:
    """Standard pinhole projection of cam-frame joints to image pixels.

    Convention:
    - input ``joints_cam`` is ``(..., 3)`` in cam-frame metres,
    - output is ``(..., 2)`` in image-frame pixels (u, v),
    - ``u = focal * X / Z + cx``, ``v = focal * Y / Z + cy``,
    - Z <= ``min_depth`` (i.e. behind/at camera) maps to NaN to avoid
      invalid 1/Z divisions and so downstream MPJPE skips that point.
    """

    def test_unit_focal_no_offset(self):
        """f=1, cx=cy=0 -> uv == (X/Z, Y/Z)."""
        j = np.array([[[1.0, 2.0, 4.0]]])
        uv = reproject_3d_to_2d(j, focal=1.0, cx=0.0, cy=0.0)
        np.testing.assert_allclose(uv[0, 0], [0.25, 0.5], atol=1e-9)

    def test_with_focal_and_principal_point(self):
        """f=100, cx=200, cy=300 on (X,Y,Z)=(1,1,2)."""
        j = np.array([[[1.0, 1.0, 2.0]]])
        uv = reproject_3d_to_2d(j, focal=100.0, cx=200.0, cy=300.0)
        np.testing.assert_allclose(uv[0, 0], [250.0, 350.0], atol=1e-9)

    def test_negative_z_returns_nan(self):
        """Joints behind the camera must NOT silently project to large pixels."""
        j = np.array([[[1.0, 1.0, -2.0], [1.0, 1.0, 5.0]]])
        uv = reproject_3d_to_2d(j, focal=100.0, cx=0.0, cy=0.0)
        assert np.isnan(uv[0, 0]).all()
        assert np.isfinite(uv[0, 1]).all()

    def test_zero_z_returns_nan(self):
        """Joints exactly at the focal plane (Z=0) project to infinity -> NaN."""
        j = np.array([[[1.0, 1.0, 0.0]]])
        uv = reproject_3d_to_2d(j, focal=100.0, cx=0.0, cy=0.0)
        assert np.isnan(uv[0, 0]).all()

    def test_propagates_nan(self):
        """NaN inputs -> NaN outputs (don't fabricate pixels for missing joints)."""
        j = np.array([[[np.nan, np.nan, np.nan], [1.0, 1.0, 2.0]]])
        uv = reproject_3d_to_2d(j, focal=100.0, cx=0.0, cy=0.0)
        assert np.isnan(uv[0, 0]).all()
        assert np.isfinite(uv[0, 1]).all()

    def test_preserves_leading_dims(self):
        """Should support arbitrary leading dims (T, J, 3) and (T, N, J, 3)."""
        rng = np.random.default_rng(0)
        j_tj = rng.standard_normal((5, 17, 3)) + np.array([0.0, 0.0, 4.0])
        j_tnj = rng.standard_normal((5, 3, 17, 3)) + np.array([0.0, 0.0, 4.0])
        uv_tj = reproject_3d_to_2d(j_tj, focal=500.0, cx=128.0, cy=128.0)
        uv_tnj = reproject_3d_to_2d(j_tnj, focal=500.0, cx=128.0, cy=128.0)
        assert uv_tj.shape == (5, 17, 2)
        assert uv_tnj.shape == (5, 3, 17, 2)


class TestMpjpe2d:
    """L2 pixel error between two ``(T, [N,] J, 2)`` arrays.

    NaN frames in either input are skipped (np.nanmean), matching the
    cam-frame ``per_joint_mpjpe`` semantics so partially-detected
    joints don't bias the result toward zero.
    """

    def test_zero_for_identical(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 1, 17, 2))
        out = mpjpe_2d(a, a.copy())
        assert out.shape == (1, 17)
        np.testing.assert_allclose(out, np.zeros((1, 17)), atol=1e-12)

    def test_constant_offset(self):
        a = np.zeros((5, 1, 17, 2))
        b = a + np.array([3.0, 4.0])
        out = mpjpe_2d(a, b)
        np.testing.assert_allclose(out, np.full((1, 17), 5.0), atol=1e-12)

    def test_nan_skipped(self):
        a = np.zeros((10, 1, 17, 2))
        b = a + np.array([3.0, 4.0])
        b[0] = np.nan
        out = mpjpe_2d(a, b)
        np.testing.assert_allclose(out, np.full((1, 17), 5.0), atol=1e-12)

    def test_shape_mismatch_raises(self):
        a = np.zeros((5, 1, 17, 2))
        b = np.zeros((5, 2, 17, 2))
        with pytest.raises(ValueError, match="shape"):
            mpjpe_2d(a, b)

    def test_returns_nan_for_dancer_with_no_finite_pairs(self):
        """A dancer with all-NaN in either input -> NaN per joint, not 0."""
        a = np.zeros((5, 2, 17, 2))
        b = np.zeros((5, 2, 17, 2))
        b[:, 1] = np.nan
        out = mpjpe_2d(a, b)
        assert np.isnan(out[1]).all()
        np.testing.assert_allclose(out[0], np.zeros(17), atol=1e-12)
