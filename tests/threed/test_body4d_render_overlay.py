"""Unit tests for :mod:`threed.sidecar_body4d.render_overlay`.

The GPU-heavy pyrender main loop runs only on the box (it needs
``trimesh`` + ``pyrender`` + EGL + the SAM-Body4D PLY artefacts), so
this module pins the **pure** helpers the loop depends on — directory
discovery, focal-JSON parsing, and the dancer-position transform that
mirrors SAM-Body4D's own renderer convention. Together these cover the
non-rasterisation edge cases so a regression in the helpers cannot
silently mis-place a dancer in the side-by-side video.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threed.sidecar_body4d.render_overlay import (
    body4d_dancer_world_pos,
    discover_body4d_dancer_ids,
    flip_yz_verts,
    load_focal_meta,
)


# ---------------------------------------------------------------------------
# discover_body4d_dancer_ids
# ---------------------------------------------------------------------------


class TestDiscoverBody4dDancerIds:
    def test_returns_sorted_int_dancers(self, tmp_path: Path):
        """Numeric subdirs are returned as sorted ints, matching SAM-Body4D's layout."""
        for d in ("3", "1", "2"):
            (tmp_path / d).mkdir()
        assert discover_body4d_dancer_ids(tmp_path) == [1, 2, 3]

    def test_skips_non_numeric_subdirs(self, tmp_path: Path):
        """SAM-Body4D never writes non-digit subdirs but we defend against it.

        If e.g. a stray ``rendered_frames`` dir lands inside
        ``mesh_4d_individual/``, we don't want it to crash the renderer
        (it would explode in ``int()``).
        """
        (tmp_path / "1").mkdir()
        (tmp_path / "2").mkdir()
        (tmp_path / "scratch").mkdir()
        (tmp_path / "_runtime").mkdir()
        assert discover_body4d_dancer_ids(tmp_path) == [1, 2]

    def test_ignores_files(self, tmp_path: Path):
        """Files in mesh_root are ignored — only subdirs count as dancers."""
        (tmp_path / "1").mkdir()
        (tmp_path / "README.md").write_text("hi", encoding="utf-8")
        assert discover_body4d_dancer_ids(tmp_path) == [1]

    def test_missing_root_returns_empty(self, tmp_path: Path):
        """A missing ``mesh_4d_individual/`` is not an error — we just have no dancers."""
        assert discover_body4d_dancer_ids(tmp_path / "does_not_exist") == []

    def test_empty_root_returns_empty(self, tmp_path: Path):
        assert discover_body4d_dancer_ids(tmp_path) == []


# ---------------------------------------------------------------------------
# load_focal_meta
# ---------------------------------------------------------------------------


class TestLoadFocalMeta:
    def test_parses_focal_and_camera(self, tmp_path: Path):
        """SAM-Body4D writes ``{focal_length, camera}`` per (dancer, frame)."""
        path = tmp_path / "00000050.json"
        path.write_text(
            json.dumps(
                {
                    "focal_length": 547.81,
                    "camera": [0.395, 1.578, 3.428],
                }
            )
        )
        focal, cam = load_focal_meta(path)
        assert isinstance(focal, float)
        assert focal == pytest.approx(547.81)
        assert cam.shape == (3,)
        assert cam.dtype == np.float32
        np.testing.assert_allclose(cam, [0.395, 1.578, 3.428], rtol=1e-5)

    def test_handles_int_focal(self, tmp_path: Path):
        path = tmp_path / "00000000.json"
        path.write_text(json.dumps({"focal_length": 600, "camera": [0, 0, 5]}))
        focal, _cam = load_focal_meta(path)
        assert isinstance(focal, float)
        assert focal == 600.0

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_focal_meta(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# body4d_dancer_world_pos
# ---------------------------------------------------------------------------


class TestBody4dDancerWorldPos:
    """Mirrors SAM-Body4D's ``camera_translation[0] *= -1`` convention.

    Upstream's :class:`Renderer` puts the camera at world position
    ``(-cam_t.x, cam_t.y, cam_t.z)`` and consumes the PLY vertices
    as-is (they're already in that "post-flip world" — saved by
    upstream's :func:`vertices_to_trimesh` as
    ``(pred_verts + cam_t) * [1, -1, -1]``). For a single pyrender
    scene with **all** dancers we move the camera to the world origin
    and translate each dancer by ``-camera_world``, i.e.
    ``+(cam_t.x, -cam_t.y, -cam_t.z)``. This helper returns that
    translation so callers can add it directly to the PLY vertices.
    """

    def test_shape_and_dtype(self):
        cam_t = np.array([0.395, 1.578, 3.428], dtype=np.float32)
        out = body4d_dancer_world_pos(cam_t)
        assert out.shape == (3,)
        assert out.dtype == np.float32

    def test_negates_yz_keeps_x(self):
        """X stays positive (same sign as cam_t.x); Y and Z get flipped."""
        cam_t = np.array([0.5, 1.5, 3.5], dtype=np.float32)
        out = body4d_dancer_world_pos(cam_t)
        np.testing.assert_allclose(out, [0.5, -1.5, -3.5])

    def test_negative_x_stays_negative(self):
        """Per SAM-Body4D's convention, the X sign is flipped twice (once when
        upstream negates ``camera.x``, once when we shift the dancer to the
        camera-at-origin frame), netting to: dancer translation X equals
        ``cam_t.x``.
        """
        cam_t = np.array([-1.639, 1.717, 3.168], dtype=np.float32)
        out = body4d_dancer_world_pos(cam_t)
        np.testing.assert_allclose(out, [-1.639, -1.717, -3.168])

    def test_zero_camera_zero_translation(self):
        """A camera at the model origin yields a zero translation —
        useful sanity check, even if SAM-Body4D never produces that."""
        out = body4d_dancer_world_pos(np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(out, np.zeros(3, dtype=np.float32))

    def test_translation_matches_negated_upstream_camera(self):
        """The returned translation must equal ``-(upstream camera_world)``
        so that ``v_PLY + translation`` puts the dancer in a frame where
        the shared camera is at the world origin.

        ``upstream_camera_world = (-cam_t.x, cam_t.y, cam_t.z)`` (per
        SAM-Body4D's :class:`Renderer`); ``-upstream`` is exactly
        ``(cam_t.x, -cam_t.y, -cam_t.z)``.
        """
        cam_t = np.array([0.7, 1.2, 4.5], dtype=np.float32)
        upstream_camera_world = np.array([-cam_t[0], cam_t[1], cam_t[2]], dtype=np.float32)
        np.testing.assert_allclose(
            body4d_dancer_world_pos(cam_t),
            -upstream_camera_world,
        )


# ---------------------------------------------------------------------------
# flip_yz_verts
# ---------------------------------------------------------------------------


class TestFlipYzVerts:
    """Flips Y/Z signs of mesh vertices — equivalent to a 180° X-rotation.

    Generic geometry helper. Not used by the body4d overlay path
    (PLYs come out of SAM-Body4D's :func:`vertices_to_trimesh`
    already 180°-X-rotated so they're consumed as-is), but kept and
    tested as a standalone utility because the same convention
    conversion is common when bridging OpenCV-style and OpenGL-style
    coordinates.
    """

    def test_shape_preserved(self):
        v = np.random.RandomState(0).randn(100, 3).astype(np.float32)
        out = flip_yz_verts(v)
        assert out.shape == v.shape
        assert out.dtype == np.float32

    def test_x_unchanged(self):
        v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = flip_yz_verts(v)
        np.testing.assert_array_equal(out[:, 0], v[:, 0])

    def test_y_negated(self):
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = flip_yz_verts(v)
        assert out[0, 1] == -2.0

    def test_z_negated(self):
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = flip_yz_verts(v)
        assert out[0, 2] == -3.0

    def test_idempotent_in_pairs(self):
        """Applying the flip twice returns the original (it's a 180° rotation)."""
        v = np.random.RandomState(1).randn(50, 3).astype(np.float32)
        np.testing.assert_array_equal(flip_yz_verts(flip_yz_verts(v)), v)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            flip_yz_verts(np.zeros((5,), dtype=np.float32))
        with pytest.raises(ValueError):
            flip_yz_verts(np.zeros((5, 4), dtype=np.float32))
