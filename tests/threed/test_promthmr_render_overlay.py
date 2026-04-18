"""Unit tests for :mod:`threed.sidecar_promthmr.render_overlay`.

The GPU-heavy mesh-rasterisation main loop runs only on the box (it
needs ``smplx`` + ``pyrender`` + a SMPL-X ``.npz`` checkpoint we don't
ship to CI), so this module pins the **pure** helpers used by that
loop — color palette, axis-angle conversion, frame indexing,
overlay compositing, intrinsics matrix. Together they cover every
non-rasterisation edge case so a regression in the helpers cannot
silently corrupt the on-box render.
"""
from __future__ import annotations

import numpy as np
import pytest

from threed.sidecar_promthmr.render_overlay import (
    DEFAULT_MESH_ALPHA,
    composite_overlay,
    dancer_color_palette,
    frame_dancer_index,
    make_intrinsics_K,
    pose_axis_angle_from_rotmat,
)


# ---------------------------------------------------------------------------
# dancer_color_palette
# ---------------------------------------------------------------------------


class TestDancerColorPalette:
    def test_shape_and_dtype(self):
        p = dancer_color_palette(5)
        assert p.shape == (5, 3)
        assert p.dtype == np.float32

    def test_values_in_unit_range(self):
        p = dancer_color_palette(20)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_colors_are_distinct(self):
        """No two dancers should share the exact same RGB triple."""
        p = dancer_color_palette(10)
        n = p.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert not np.allclose(p[i], p[j]), f"dancers {i} and {j} share color"

    def test_deterministic(self):
        """Two calls with the same n must return the identical palette.

        The orchestrator may call the renderer multiple times (e.g. a
        re-stitch after a tweak); we don't want dancer colors to flicker
        between runs.
        """
        a = dancer_color_palette(7)
        b = dancer_color_palette(7)
        np.testing.assert_array_equal(a, b)

    def test_zero_returns_empty(self):
        p = dancer_color_palette(0)
        assert p.shape == (0, 3)


# ---------------------------------------------------------------------------
# pose_axis_angle_from_rotmat
# ---------------------------------------------------------------------------


class TestPoseAxisAngleFromRotmat:
    def test_identity_rotmat_yields_zero_aa(self):
        """Identity rotation matrix -> zero axis-angle vector."""
        B, J = 3, 55
        rotmat = np.tile(np.eye(3, dtype=np.float32)[None, None], (B, J, 1, 1))
        aa = pose_axis_angle_from_rotmat(rotmat)
        assert aa.shape == (B, J * 3)
        assert aa.dtype == np.float32
        np.testing.assert_allclose(aa, 0.0, atol=1e-7)

    def test_known_rotation_about_z(self):
        """90-deg rotation about z -> axis-angle = (0, 0, pi/2)."""
        theta = np.pi / 2
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ], dtype=np.float32)
        rotmat = Rz[None, None]  # (B=1, J=1, 3, 3)
        aa = pose_axis_angle_from_rotmat(rotmat)
        np.testing.assert_allclose(aa[0], [0.0, 0.0, np.pi / 2], atol=1e-6)

    def test_round_trip_on_random_rotations(self):
        """rotvec -> matrix -> rotvec round-trips to the same vector."""
        from scipy.spatial.transform import Rotation as Rsp
        rng = np.random.default_rng(42)
        rotvec = rng.standard_normal((4, 55, 3)).astype(np.float32) * 0.5
        rotmat = Rsp.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
        rotmat = rotmat.reshape(4, 55, 3, 3).astype(np.float32)
        aa = pose_axis_angle_from_rotmat(rotmat)
        np.testing.assert_allclose(aa.reshape(4, 55, 3), rotvec, atol=1e-5)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            pose_axis_angle_from_rotmat(np.eye(3, dtype=np.float32))

    def test_rejects_wrong_trailing_shape(self):
        with pytest.raises(ValueError):
            pose_axis_angle_from_rotmat(np.zeros((1, 1, 4, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# make_intrinsics_K
# ---------------------------------------------------------------------------


class TestIntrinsicsMatrix:
    def test_basic(self):
        K = make_intrinsics_K(1280.0, 640.0, 360.0)
        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        np.testing.assert_allclose(
            K,
            [[1280.0, 0.0, 640.0],
             [0.0, 1280.0, 360.0],
             [0.0, 0.0, 1.0]],
        )

    def test_off_center_principal_point(self):
        K = make_intrinsics_K(900.0, 100.0, 200.0)
        assert K[0, 2] == pytest.approx(100.0)
        assert K[1, 2] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# frame_dancer_index
# ---------------------------------------------------------------------------


class TestFrameDancerIndex:
    def test_full_clip_all_dancers(self):
        """All dancers present every frame -> (di, local_idx) pair per frame."""
        per_dancer_frames = [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ]
        idx = frame_dancer_index(per_dancer_frames, n_frames=3)
        assert sorted(idx[0]) == [(0, 0), (1, 0)]
        assert sorted(idx[1]) == [(0, 1), (1, 1)]
        assert sorted(idx[2]) == [(0, 2), (1, 2)]

    def test_partial_track(self):
        """Dancer 1 only appears on frames 1, 2 (gap on frame 0)."""
        per_dancer_frames = [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        ]
        idx = frame_dancer_index(per_dancer_frames, n_frames=3)
        assert idx[0] == [(0, 0)]
        assert sorted(idx[1]) == [(0, 1), (1, 0)]
        assert sorted(idx[2]) == [(0, 2), (1, 1)]

    def test_empty_frames(self):
        """Frames where no dancer is present return an empty list."""
        per_dancer_frames = [np.array([2], dtype=np.int64)]
        idx = frame_dancer_index(per_dancer_frames, n_frames=3)
        assert idx[0] == []
        assert idx[1] == []
        assert idx[2] == [(0, 0)]

    def test_returns_one_entry_per_frame(self):
        per_dancer_frames = [np.array([0, 1], dtype=np.int64)]
        idx = frame_dancer_index(per_dancer_frames, n_frames=5)
        assert len(idx) == 5

    def test_ignores_out_of_range_frames(self):
        """Frames >= n_frames are silently dropped (PHMR may pad)."""
        per_dancer_frames = [np.array([0, 1, 99], dtype=np.int64)]
        idx = frame_dancer_index(per_dancer_frames, n_frames=2)
        assert idx[0] == [(0, 0)]
        assert idx[1] == [(0, 1)]


# ---------------------------------------------------------------------------
# composite_overlay
# ---------------------------------------------------------------------------


class TestCompositeOverlay:
    def test_zero_alpha_returns_input(self):
        rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        rendered = np.full((4, 4, 3), 200, dtype=np.uint8)
        alpha = np.zeros((4, 4), dtype=np.float32)
        out = composite_overlay(rgb, rendered, alpha)
        np.testing.assert_array_equal(out, rgb)

    def test_full_alpha_returns_render(self):
        rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        rendered = np.full((4, 4, 3), 200, dtype=np.uint8)
        alpha = np.ones((4, 4), dtype=np.float32)
        out = composite_overlay(rgb, rendered, alpha)
        np.testing.assert_array_equal(out, rendered)

    def test_half_alpha_blends(self):
        """alpha=0.5 -> (input + render) / 2."""
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        rendered = np.full((2, 2, 3), 200, dtype=np.uint8)
        alpha = np.full((2, 2), 0.5, dtype=np.float32)
        out = composite_overlay(rgb, rendered, alpha)
        np.testing.assert_array_equal(out, np.full((2, 2, 3), 150, dtype=np.uint8))

    def test_per_pixel_alpha(self):
        """Spatially varying alpha blends pixel-by-pixel."""
        rgb = np.full((1, 4, 3), 0, dtype=np.uint8)
        rendered = np.full((1, 4, 3), 240, dtype=np.uint8)
        alpha = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)
        out = composite_overlay(rgb, rendered, alpha)
        np.testing.assert_array_equal(out[0, :, 0], [0, 60, 120, 240])

    def test_dtype_is_uint8(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rendered = np.zeros((2, 2, 3), dtype=np.uint8)
        alpha = np.zeros((2, 2), dtype=np.float32)
        out = composite_overlay(rgb, rendered, alpha)
        assert out.dtype == np.uint8

    def test_shape_mismatch_raises(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rendered = np.zeros((4, 4, 3), dtype=np.uint8)
        alpha = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            composite_overlay(rgb, rendered, alpha)

    def test_default_mesh_alpha_in_unit_range(self):
        """The render uses ``DEFAULT_MESH_ALPHA`` for every fragment; pin it
        in [0, 1] so a misconfig can't blow blending up."""
        assert 0.0 <= DEFAULT_MESH_ALPHA <= 1.0
