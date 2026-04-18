"""Unit tests for :mod:`threed.compare.run_compare` (plan Task 11d)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from threed.compare.run_compare import (
    align_arrays,
    auto_reduce_to_coco17,
    main,
)


@pytest.fixture
def coco17_a(tmp_path: Path) -> Path:
    j = np.zeros((10, 2, 17, 3), dtype=np.float32)
    p = tmp_path / "phmr_joints.npy"
    np.save(p, j)
    return p


@pytest.fixture
def coco17_b(tmp_path: Path) -> Path:
    j = np.zeros((10, 2, 17, 3), dtype=np.float32)
    j[:, :, :, 0] = 0.25
    p = tmp_path / "body4d_joints.npy"
    np.save(p, j)
    return p


class TestAlignArrays:
    def test_truncates_to_min_frames(self):
        a = np.zeros((10, 2, 17, 3), dtype=np.float32)
        b = np.zeros((8, 2, 17, 3), dtype=np.float32)
        a2, b2 = align_arrays(a, b)
        assert a2.shape == (8, 2, 17, 3)
        assert b2.shape == (8, 2, 17, 3)

    def test_truncates_to_min_dancers(self):
        a = np.zeros((5, 4, 17, 3), dtype=np.float32)
        b = np.zeros((5, 2, 17, 3), dtype=np.float32)
        a2, b2 = align_arrays(a, b)
        assert a2.shape == (5, 2, 17, 3)
        assert b2.shape == (5, 2, 17, 3)

    def test_no_op_when_already_aligned(self):
        a = np.zeros((5, 3, 17, 3), dtype=np.float32)
        b = np.zeros((5, 3, 17, 3), dtype=np.float32)
        a2, b2 = align_arrays(a, b)
        assert a2.shape == b2.shape == (5, 3, 17, 3)

    def test_joint_axis_mismatch_raises(self):
        a = np.zeros((5, 3, 22, 3), dtype=np.float32)
        b = np.zeros((5, 3, 17, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="joint"):
            align_arrays(a, b)


class TestAutoReduceToCoco17:
    def test_no_op_when_already_17(self):
        j = np.zeros((5, 1, 17, 3), dtype=np.float32)
        out = auto_reduce_to_coco17(j)
        assert out.shape == (5, 1, 17, 3)

    def test_reduces_smpl22(self):
        j = np.zeros((5, 1, 22, 3), dtype=np.float32)
        out = auto_reduce_to_coco17(j)
        assert out.shape == (5, 1, 17, 3)

    def test_reduces_mhr70(self):
        j = np.zeros((5, 1, 70, 3), dtype=np.float32)
        out = auto_reduce_to_coco17(j)
        assert out.shape == (5, 1, 17, 3)

    def test_unknown_size_raises(self):
        j = np.zeros((5, 1, 25, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="25"):
            auto_reduce_to_coco17(j)


class TestMainEndToEnd:
    def test_writes_metrics_json(self, coco17_a, coco17_b, tmp_path):
        out = tmp_path / "metrics.json"
        rc = main([
            "--prompthmr-joints", str(coco17_a),
            "--body4d-joints", str(coco17_b),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.is_file()

        m = json.loads(out.read_text())
        assert m["n_frames_compared"] == 10
        assert m["n_dancers_compared"] == 2
        assert "per_joint_jitter_phmr_m_per_frame" in m
        assert "per_joint_jitter_body4d_m_per_frame" in m
        assert "per_joint_mpjpe_m" in m
        assert "foot_skating_phmr_m_per_frame" in m
        assert "foot_skating_body4d_m_per_frame" in m

        mpjpe = np.array(m["per_joint_mpjpe_m"])
        np.testing.assert_allclose(mpjpe, np.full((2, 17), 0.25), rtol=1e-5)

    def test_creates_output_parent_dir(self, coco17_a, coco17_b, tmp_path):
        out = tmp_path / "deeply" / "nested" / "metrics.json"
        rc = main([
            "--prompthmr-joints", str(coco17_a),
            "--body4d-joints", str(coco17_b),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.is_file()

    def test_truncates_dancer_mismatch(self, tmp_path):
        a = np.zeros((10, 4, 17, 3), dtype=np.float32)
        b = np.zeros((10, 2, 17, 3), dtype=np.float32)
        ap = tmp_path / "a.npy"
        bp = tmp_path / "b.npy"
        np.save(ap, a)
        np.save(bp, b)
        out = tmp_path / "m.json"
        rc = main([
            "--prompthmr-joints", str(ap),
            "--body4d-joints", str(bp),
            "--output", str(out),
        ])
        assert rc == 0
        m = json.loads(out.read_text())
        assert m["n_dancers_compared"] == 2
        assert m["n_dancers_phmr"] == 4
        assert m["n_dancers_body4d"] == 2

    def test_emits_pa_mpjpe_field(self, tmp_path):
        """Followup #1: metrics.json must include Procrustes-aligned MPJPE.

        Constructed input: phmr is at origin, body4d is the same poses
        translated by a constant offset per dancer. Raw MPJPE ~ |offset|.
        PA-MPJPE (per-dancer alignment removes the translation) ~ 0.
        """
        rng = np.random.default_rng(123)
        a = rng.standard_normal((30, 2, 17, 3)).astype(np.float32)
        b = a.copy()
        b[:, 0] += np.array([3.0, 0.0, 4.0], dtype=np.float32)
        b[:, 1] += np.array([0.0, 5.0, -12.0], dtype=np.float32)
        ap = tmp_path / "a.npy"
        bp = tmp_path / "b.npy"
        np.save(ap, a)
        np.save(bp, b)
        out = tmp_path / "m.json"
        rc = main([
            "--prompthmr-joints", str(ap),
            "--body4d-joints", str(bp),
            "--output", str(out),
        ])
        assert rc == 0
        m = json.loads(out.read_text())
        assert "per_joint_mpjpe_pa_m" in m, "PA-MPJPE field missing from metrics.json"
        raw = np.array(m["per_joint_mpjpe_m"])
        pa = np.array(m["per_joint_mpjpe_pa_m"])
        assert raw.shape == (2, 17)
        assert pa.shape == (2, 17)
        assert float(raw.mean()) > 5.0
        assert float(pa.mean()) < 1e-4

    def test_emits_world_frame_foot_skating_field_when_provided(self, tmp_path, coco17_a, coco17_b):
        """Followup #2: when ``--prompthmr-world-joints`` is provided,
        ``metrics.json`` gains a ``foot_skating_phmr_world_m_per_frame``
        field computed from PHMR's SMPL-22 world joints. Without the flag,
        the field is absent (back-compat)."""
        world = np.zeros((10, 2, 22, 3), dtype=np.float32)
        world[:, 0, 7, 0] = np.arange(10) * 0.1
        wp = tmp_path / "world.npy"
        np.save(wp, world)
        out = tmp_path / "m.json"
        rc = main([
            "--prompthmr-joints", str(coco17_a),
            "--body4d-joints", str(coco17_b),
            "--output", str(out),
            "--prompthmr-world-joints", str(wp),
        ])
        assert rc == 0
        m = json.loads(out.read_text())
        assert "foot_skating_phmr_world_m_per_frame" in m
        ws = np.array(m["foot_skating_phmr_world_m_per_frame"])
        assert ws.shape == (2,)
        np.testing.assert_allclose(ws[0], 0.1, rtol=1e-5)
        assert ws[1] == 0.0

    def test_world_frame_foot_skating_absent_when_world_joints_missing(self, tmp_path, coco17_a, coco17_b):
        """Back-compat: omitting ``--prompthmr-world-joints`` -> field is not present."""
        out = tmp_path / "m.json"
        rc = main([
            "--prompthmr-joints", str(coco17_a),
            "--body4d-joints", str(coco17_b),
            "--output", str(out),
        ])
        assert rc == 0
        m = json.loads(out.read_text())
        assert "foot_skating_phmr_world_m_per_frame" not in m
