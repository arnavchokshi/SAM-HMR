"""Unit tests for :mod:`threed.sidecar_promthmr.project_joints` (plan Task 11f)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from threed.sidecar_promthmr.project_joints import (
    project_joints_world_to_cam,
    main,
)


class TestProjectJointsWorldToCam:
    def test_identity_extrinsics(self):
        """R=I, T=0 -> joints unchanged."""
        joints_world = np.array([
            [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]],
            [[2.0, 4.0, 6.0], [1.0, 1.0, 1.0]],
        ], dtype=np.float32).reshape(2, 1, 2, 3)
        Rcw = np.tile(np.eye(3, dtype=np.float32)[None], (2, 1, 1))
        Tcw = np.zeros((2, 3), dtype=np.float32)
        out = project_joints_world_to_cam(joints_world, Rcw, Tcw)
        np.testing.assert_allclose(out, joints_world)

    def test_pure_translation(self):
        """R=I, T=(1,0,0) -> joints shifted by (1,0,0) per frame."""
        joints_world = np.zeros((1, 1, 1, 3), dtype=np.float32)
        Rcw = np.eye(3, dtype=np.float32)[None]
        Tcw = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = project_joints_world_to_cam(joints_world, Rcw, Tcw)
        np.testing.assert_allclose(out[0, 0, 0], [1.0, 0.0, 0.0])

    def test_pure_rotation(self):
        """90 deg about z -> (1, 0, 0) -> (0, 1, 0)."""
        joints_world = np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.float32)
        theta = np.pi / 2
        Rcw = np.array([[[
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ]]], dtype=np.float32).reshape(1, 3, 3)
        Tcw = np.zeros((1, 3), dtype=np.float32)
        out = project_joints_world_to_cam(joints_world, Rcw, Tcw)
        np.testing.assert_allclose(out[0, 0, 0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_propagates_nan(self):
        joints_world = np.full((2, 1, 1, 3), np.nan, dtype=np.float32)
        Rcw = np.tile(np.eye(3)[None], (2, 1, 1)).astype(np.float32)
        Tcw = np.zeros((2, 3), dtype=np.float32)
        out = project_joints_world_to_cam(joints_world, Rcw, Tcw)
        assert np.isnan(out).all()

    def test_per_frame_extrinsics_independent(self):
        """Frame 0 with T=(1,0,0), frame 1 with T=(0,2,0)."""
        joints_world = np.zeros((2, 1, 1, 3), dtype=np.float32)
        Rcw = np.tile(np.eye(3)[None], (2, 1, 1)).astype(np.float32)
        Tcw = np.array([[1.0, 0, 0], [0, 2.0, 0]], dtype=np.float32)
        out = project_joints_world_to_cam(joints_world, Rcw, Tcw)
        np.testing.assert_allclose(out[0, 0, 0], [1, 0, 0])
        np.testing.assert_allclose(out[1, 0, 0], [0, 2, 0])

    def test_shape_mismatch_raises(self):
        joints_world = np.zeros((10, 2, 22, 3), dtype=np.float32)
        Rcw = np.tile(np.eye(3)[None], (8, 1, 1)).astype(np.float32)
        Tcw = np.zeros((8, 3), dtype=np.float32)
        with pytest.raises(ValueError, match=r"frame|10|8"):
            project_joints_world_to_cam(joints_world, Rcw, Tcw)


class TestMainEndToEnd:
    def test_writes_coco17_in_cam_frame(self, tmp_path: Path):
        """results.pkl + joints_world.npy -> joints_coco17_cam.npy.

        Builds a minimal results.pkl with ``camera_world.Rcw`` /
        ``Tcw`` and a SMPL-22 joints_world.npy, then verifies the
        output file exists, has the expected shape, and matches the
        composition (project + reduce).
        """
        import joblib

        prompthmr_dir = tmp_path / "prompthmr"
        prompthmr_dir.mkdir()

        T, N = 5, 2
        rng = np.random.default_rng(0)
        joints_world = rng.standard_normal((T, N, 22, 3)).astype(np.float32)
        np.save(prompthmr_dir / "joints_world.npy", joints_world)

        Tcw = rng.standard_normal((T, 3)).astype(np.float32)
        Rcw = np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1))
        results = {
            "camera_world": {
                "Rcw": Rcw,
                "Tcw": Tcw,
                "img_focal": 1000,
            },
        }
        joblib.dump(results, prompthmr_dir / "results.pkl")

        out = prompthmr_dir / "joints_coco17_cam.npy"
        rc = main([
            "--prompthmr-dir", str(prompthmr_dir),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.is_file()
        out_arr = np.load(out)
        assert out_arr.shape == (T, N, 17, 3)
        from threed.compare.joints import SMPL22_TO_COCO17
        expected = (joints_world + Tcw[:, None, None, :])[..., SMPL22_TO_COCO17, :]
        np.testing.assert_allclose(out_arr, expected, rtol=1e-5, atol=1e-5)

    def test_missing_results_pkl_errors(self, tmp_path: Path):
        prompthmr_dir = tmp_path / "prompthmr"
        prompthmr_dir.mkdir()
        np.save(prompthmr_dir / "joints_world.npy",
                np.zeros((1, 1, 22, 3), dtype=np.float32))
        rc = main([
            "--prompthmr-dir", str(prompthmr_dir),
            "--output", str(tmp_path / "out.npy"),
        ])
        assert rc != 0

    def test_missing_joints_world_errors(self, tmp_path: Path):
        import joblib
        prompthmr_dir = tmp_path / "prompthmr"
        prompthmr_dir.mkdir()
        joblib.dump({"camera_world": {
            "Rcw": np.eye(3)[None].astype(np.float32),
            "Tcw": np.zeros((1, 3), dtype=np.float32),
        }}, prompthmr_dir / "results.pkl")
        rc = main([
            "--prompthmr-dir", str(prompthmr_dir),
            "--output", str(tmp_path / "out.npy"),
        ])
        assert rc != 0
