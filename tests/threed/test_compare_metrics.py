"""Unit tests for :mod:`threed.compare.metrics` (plan Task 11b)."""
from __future__ import annotations

import numpy as np
import pytest

from threed.compare.metrics import (
    foot_skating,
    per_joint_jitter,
    per_joint_mpjpe,
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
