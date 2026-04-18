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
    discover_body4d_dancer_ids,
    flip_yz_verts,
    load_focal_meta,
    upstream_ply_centroid,
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
# upstream_ply_centroid
# ---------------------------------------------------------------------------


class TestUpstreamPlyCentroid:
    """Pins SAM-Body4D's PLY-on-disk convention against accidental drift.

    SAM-Body4D's :func:`save_mesh_results` (in
    ``models/sam_3d_body/notebook/utils.py``) calls
    ``Renderer.vertices_to_trimesh(pred_vertices, pred_cam_t)``; that
    function adds ``cam_t`` to the model-space vertices **then**
    applies a 180° X-rotation, i.e. emits
    ``(pred_vertices + cam_t) * [1, -1, -1]``. The multi-dancer
    rendering path (:func:`Renderer.render_rgba_multiple`) keeps the
    camera at the world origin and feeds those PLY vertices straight
    into the scene with **no extra translation**.

    Our :mod:`render_overlay` mirrors that convention, so the function
    under test deliberately models the upstream centroid math without
    any per-dancer fudge factor — if a future change tries to add a
    "world position" offset the way an earlier draft did, this suite
    will fail loudly and the side-by-side video will keep its proper
    scale.
    """

    def test_zero_pred_verts_centroid_equals_flipped_cam_t(self):
        """For ``pred_verts == 0`` the PLY centroid is ``(cam_t.x, -cam_t.y, -cam_t.z)``.

        (The X sign is **kept** — only Y/Z are flipped — because the
        multi-dancer path never negates ``cam_t.x``. The single-dancer
        path does, but we don't use that one here.)
        """
        cam_t = np.array([0.395, 1.578, 3.428], dtype=np.float32)
        c = upstream_ply_centroid(cam_t)
        np.testing.assert_allclose(c, [0.395, -1.578, -3.428])

    def test_negative_x_stays_negative(self):
        cam_t = np.array([-1.639, 1.717, 3.168], dtype=np.float32)
        c = upstream_ply_centroid(cam_t)
        np.testing.assert_allclose(c, [-1.639, -1.717, -3.168])

    def test_zero_cam_t_zero_centroid(self):
        c = upstream_ply_centroid(np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(c, np.zeros(3, dtype=np.float32))

    def test_centroid_z_is_in_front_of_origin_camera(self):
        """For typical positive-depth ``cam_t.z``, the centroid Z is negative
        — i.e. **in front of** an OpenGL camera at the world origin
        (which looks down ``-Z``). This is the property that keeps
        the mesh visible without any extra translation.
        """
        cam_t = np.array([0.0, 0.2, 5.0], dtype=np.float32)
        c = upstream_ply_centroid(cam_t)
        assert c[2] < 0
        assert c[2] == pytest.approx(-5.0)

    def test_shape_and_dtype(self):
        cam_t = np.array([0.395, 1.578, 3.428], dtype=np.float32)
        c = upstream_ply_centroid(cam_t)
        assert c.shape == (3,)
        assert c.dtype == np.float32

    def test_pred_verts_centroid_at_origin_falls_back_to_cam_t_only(self):
        """The helper takes only ``cam_t`` because the *centroid* of
        ``pred_vertices`` is approximately the model-space origin for
        a centred body mesh; this is exactly what makes multi-dancer
        compositing depth-correct without any per-dancer offset.
        """
        for cam_t in [
            np.array([0.0, 0.0, 5.0], dtype=np.float32),
            np.array([0.5, 1.0, 7.5], dtype=np.float32),
            np.array([-0.7, 1.2, 4.5], dtype=np.float32),
        ]:
            c = upstream_ply_centroid(cam_t)
            assert c[0] == pytest.approx(cam_t[0])
            assert c[1] == pytest.approx(-cam_t[1])
            assert c[2] == pytest.approx(-cam_t[2])

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            upstream_ply_centroid(np.zeros(2, dtype=np.float32))
        with pytest.raises(ValueError):
            upstream_ply_centroid(np.zeros((1, 3), dtype=np.float32))


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
