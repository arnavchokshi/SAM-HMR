"""Unit tests for :mod:`threed.sidecar_promthmr.reproject_vs_vitpose` (Followup #4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threed.sidecar_promthmr.reproject_vs_vitpose import (
    load_vitpose_padded,
    main,
)


class TestLoadVitposePadded:
    def test_pads_missing_frames_with_nan(self):
        """Track present on frames 0,2 -> frame 1 should be all NaN."""
        people = {
            7: {
                "frames": np.array([0, 2], dtype=np.int64),
                "vitpose": np.array([
                    [[1.0, 2.0, 0.9]] * 17,
                    [[5.0, 6.0, 0.8]] * 17,
                ], dtype=np.float64),
            }
        }
        out = load_vitpose_padded(people, n_frames=3, n_dancers=1, sorted_tids=[7])
        assert out.shape == (3, 1, 17, 3)
        np.testing.assert_allclose(out[0, 0, 0], [1.0, 2.0, 0.9])
        assert np.isnan(out[1, 0]).all()
        np.testing.assert_allclose(out[2, 0, 0], [5.0, 6.0, 0.8])

    def test_dancer_slot_follows_sorted_tid_order(self):
        """sorted_tids=[3,7] -> tid 3 in slot 0, tid 7 in slot 1."""
        people = {
            3: {
                "frames": np.array([0], dtype=np.int64),
                "vitpose": np.array([[[100.0, 200.0, 0.5]] * 17], dtype=np.float64),
            },
            7: {
                "frames": np.array([0], dtype=np.int64),
                "vitpose": np.array([[[10.0, 20.0, 0.5]] * 17], dtype=np.float64),
            },
        }
        out = load_vitpose_padded(people, n_frames=1, n_dancers=2, sorted_tids=[3, 7])
        np.testing.assert_allclose(out[0, 0, 0], [100.0, 200.0, 0.5])
        np.testing.assert_allclose(out[0, 1, 0], [10.0, 20.0, 0.5])

    def test_missing_tid_in_people_yields_nan_dancer(self):
        """If sorted_tids includes a tid not in people, that slot is all-NaN."""
        people = {
            7: {
                "frames": np.array([0], dtype=np.int64),
                "vitpose": np.array([[[1.0, 2.0, 0.9]] * 17], dtype=np.float64),
            },
        }
        out = load_vitpose_padded(people, n_frames=1, n_dancers=2, sorted_tids=[7, 99])
        np.testing.assert_allclose(out[0, 0, 0], [1.0, 2.0, 0.9])
        assert np.isnan(out[0, 1]).all()

    def test_out_of_range_frame_indices_ignored(self):
        """Frame indices >= n_frames or < 0 must be silently skipped."""
        people = {
            7: {
                "frames": np.array([-1, 0, 5], dtype=np.int64),
                "vitpose": np.array([
                    [[1.0, 2.0, 0.9]] * 17,
                    [[3.0, 4.0, 0.7]] * 17,
                    [[5.0, 6.0, 0.5]] * 17,
                ], dtype=np.float64),
            }
        }
        out = load_vitpose_padded(people, n_frames=3, n_dancers=1, sorted_tids=[7])
        np.testing.assert_allclose(out[0, 0, 0], [3.0, 4.0, 0.7])
        assert np.isnan(out[1, 0]).all()
        assert np.isnan(out[2, 0]).all()


class TestMainEndToEnd:
    @pytest.fixture
    def fake_phmr_dir(self, tmp_path: Path) -> Path:
        """Build a minimal results.pkl + joints_coco17_cam.npy that
        mimics PromptHMR-Vid's output for 2 frames, 1 dancer, COCO-17.
        joblib is required since real PHMR uses joblib.dump."""
        joblib = pytest.importorskip("joblib")
        d = tmp_path / "phmr"
        d.mkdir()
        n_frames, n_dancers = 2, 1
        coco = np.zeros((n_frames, n_dancers, 17, 3), dtype=np.float32)
        coco[..., 2] = 4.0
        coco[..., 0] = 1.0
        coco[..., 1] = 1.0
        np.save(d / "joints_coco17_cam.npy", coco)
        results = {
            "camera": {
                "img_focal": np.int64(100),
                "img_center": [50.0, 60.0],
            },
            "people": {
                7: {
                    "frames": np.array([0, 1], dtype=np.int64),
                    "vitpose": np.tile(
                        np.array([[[75.0, 85.0, 0.9]]], dtype=np.float64),
                        (2, 17, 1),
                    ),
                },
            },
        }
        joblib.dump(results, d / "results.pkl")
        return d

    def test_writes_reproj_metrics_json(self, fake_phmr_dir, tmp_path):
        """Per-joint pixel MPJPE between PHMR reproj and ViTPose (PHMR coords)."""
        out = tmp_path / "reproj.json"
        rc = main([
            "--prompthmr-dir", str(fake_phmr_dir),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.is_file()
        m = json.loads(out.read_text())
        assert m["n_frames"] == 2
        assert m["n_dancers"] == 1
        assert "per_joint_mpjpe_phmr_vs_vitpose_px" in m
        pjpe = np.array(m["per_joint_mpjpe_phmr_vs_vitpose_px"])
        assert pjpe.shape == (1, 17)
        np.testing.assert_allclose(pjpe, np.zeros((1, 17)), atol=1e-9)
        assert m["phmr_focal"] == 100.0
        assert m["phmr_cx"] == 50.0
        assert m["phmr_cy"] == 60.0

    def test_low_confidence_vitpose_excluded(self, fake_phmr_dir, tmp_path):
        """ViTPose detections with conf < threshold are masked as NaN."""
        joblib = pytest.importorskip("joblib")
        results = joblib.load(fake_phmr_dir / "results.pkl")
        results["people"][7]["vitpose"][0, :, 2] = 0.05
        joblib.dump(results, fake_phmr_dir / "results.pkl")
        out = tmp_path / "reproj.json"
        rc = main([
            "--prompthmr-dir", str(fake_phmr_dir),
            "--output", str(out),
            "--vitpose-conf-threshold", "0.5",
        ])
        assert rc == 0
        m = json.loads(out.read_text())
        assert m["vitpose_conf_threshold"] == 0.5
        assert m["n_low_confidence_keypoints"] >= 17

    def test_creates_output_parent_dir(self, fake_phmr_dir, tmp_path):
        out = tmp_path / "deeply" / "nested" / "reproj.json"
        rc = main([
            "--prompthmr-dir", str(fake_phmr_dir),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.is_file()
