"""Unit tests for :mod:`threed.sidecar_body4d.reproject_vs_vitpose`.

Body4D-vs-ViTPose 2-D reprojection (Followup #4 — Body4D side, addresses
the user's correct observation that "PHMR is smoother in 3D" doesn't
mean "PHMR is more accurate visually"). Compares Body4D's projected 3-D
joints against ViTPose detections from PHMR's bundled output, with
ViTPose coords scaled from PHMR's canvas (e.g. 504×896 portrait) back
into the native frame resolution (e.g. 576×1024 portrait) so the two
pipelines can be measured on the same coordinate system.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from threed.sidecar_body4d.reproject_vs_vitpose import (
    body4d_joints_to_image_2d,
    load_body4d_focal_cam_t_per_frame,
    main,
    read_native_frame_size,
    scale_vitpose_to_native,
)


# ---------------------------------------------------------------------------
# read_native_frame_size
# ---------------------------------------------------------------------------


class TestReadNativeFrameSize:
    def test_returns_wh_from_first_jpg(self, tmp_path):
        d = tmp_path / "frames_full"
        d.mkdir()
        Image.new("RGB", (1280, 720), color=(0, 0, 0)).save(d / "00000000.jpg")
        Image.new("RGB", (1280, 720), color=(0, 0, 0)).save(d / "00000001.jpg")
        assert read_native_frame_size(d) == (1280, 720)

    def test_handles_portrait(self, tmp_path):
        d = tmp_path / "frames"
        d.mkdir()
        Image.new("RGB", (576, 1024), color=(0, 0, 0)).save(d / "00000000.jpg")
        assert read_native_frame_size(d) == (576, 1024)

    def test_raises_when_empty(self, tmp_path):
        d = tmp_path / "frames"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            read_native_frame_size(d)


# ---------------------------------------------------------------------------
# load_body4d_focal_cam_t_per_frame
# ---------------------------------------------------------------------------


class TestLoadBody4dFocalCamT:
    def test_loads_dense_focal_and_camt_arrays(self, tmp_path):
        focal_dir = tmp_path / "focal_4d_individual"
        pid_dir = focal_dir / "1"
        pid_dir.mkdir(parents=True)
        for t in range(3):
            (pid_dir / f"{t:08d}.json").write_text(json.dumps({
                "focal_length": 555.0 + t,
                "camera": [0.1 * t, 0.2 * t, 3.0 + t],
            }))
        focals, cam_ts = load_body4d_focal_cam_t_per_frame(focal_dir, pid=1, n_frames=3)
        assert focals.shape == (3,)
        assert cam_ts.shape == (3, 3)
        np.testing.assert_allclose(focals, [555.0, 556.0, 557.0])
        np.testing.assert_allclose(cam_ts[2], [0.2, 0.4, 5.0])

    def test_missing_frame_yields_nan(self, tmp_path):
        focal_dir = tmp_path / "focal_4d_individual"
        pid_dir = focal_dir / "2"
        pid_dir.mkdir(parents=True)
        (pid_dir / "00000000.json").write_text(json.dumps(
            {"focal_length": 555.0, "camera": [0.0, 0.0, 3.0]}
        ))
        focals, cam_ts = load_body4d_focal_cam_t_per_frame(focal_dir, pid=2, n_frames=2)
        assert np.isnan(focals[1])
        assert np.isnan(cam_ts[1]).all()

    def test_missing_pid_dir_yields_all_nan(self, tmp_path):
        focal_dir = tmp_path / "focal_4d_individual"
        focal_dir.mkdir()
        focals, cam_ts = load_body4d_focal_cam_t_per_frame(focal_dir, pid=99, n_frames=4)
        assert focals.shape == (4,)
        assert cam_ts.shape == (4, 3)
        assert np.isnan(focals).all()
        assert np.isnan(cam_ts).all()


# ---------------------------------------------------------------------------
# body4d_joints_to_image_2d
# ---------------------------------------------------------------------------


class TestBody4dJointsToImage2d:
    def test_simple_projection(self):
        # One frame, one dancer: joint at (0.5, 0.0, 0.0) local + cam_t (0, 0, 5)
        # => cam frame (0.5, 0, 5) => u = focal*0.5/5 + cx = 555*0.1+640 = 695.5
        joints_local = np.zeros((1, 1, 70, 3), dtype=np.float64)
        joints_local[0, 0, 0] = [0.5, 0.0, 0.0]
        focals = np.array([[555.0]], dtype=np.float64)  # (T, N)
        cam_ts = np.zeros((1, 1, 3), dtype=np.float64)
        cam_ts[0, 0] = [0.0, 0.0, 5.0]
        out = body4d_joints_to_image_2d(
            joints_local, focals=focals, cam_ts=cam_ts,
            cx=640.0, cy=360.0,
            joint_index_subset=[0] * 17,
        )
        assert out.shape == (1, 1, 17, 2)
        np.testing.assert_allclose(out[0, 0, 0], [695.5, 360.0], atol=1e-6)

    def test_negative_z_yields_nan(self):
        joints_local = np.zeros((1, 1, 70, 3), dtype=np.float64)
        joints_local[0, 0, 0] = [0.5, 0.0, 0.0]
        focals = np.array([[555.0]], dtype=np.float64)
        cam_ts = np.zeros((1, 1, 3), dtype=np.float64)
        cam_ts[0, 0] = [0.0, 0.0, -1.0]  # behind camera
        out = body4d_joints_to_image_2d(
            joints_local, focals=focals, cam_ts=cam_ts,
            cx=640.0, cy=360.0,
            joint_index_subset=[0] * 17,
        )
        assert np.isnan(out[0, 0, 0]).all()

    def test_nan_focal_yields_nan_dancer(self):
        joints_local = np.zeros((1, 1, 70, 3), dtype=np.float64)
        focals = np.array([[np.nan]], dtype=np.float64)
        cam_ts = np.zeros((1, 1, 3), dtype=np.float64)
        cam_ts[0, 0] = [0.0, 0.0, 5.0]
        out = body4d_joints_to_image_2d(
            joints_local, focals=focals, cam_ts=cam_ts,
            cx=640.0, cy=360.0,
            joint_index_subset=[0] * 17,
        )
        assert np.isnan(out).all()

    def test_uses_actual_mhr70_to_coco17_subset(self):
        # Different joints -> different 2D output (sanity that subset is applied)
        joints_local = np.zeros((1, 1, 70, 3), dtype=np.float64)
        joints_local[0, 0, 5] = [1.0, 0.0, 0.0]  # MHR joint 5 has X=1
        joints_local[0, 0, 6] = [-1.0, 0.0, 0.0]  # MHR joint 6 has X=-1
        focals = np.array([[100.0]], dtype=np.float64)
        cam_ts = np.zeros((1, 1, 3), dtype=np.float64)
        cam_ts[0, 0] = [0.0, 0.0, 10.0]
        out = body4d_joints_to_image_2d(
            joints_local, focals=focals, cam_ts=cam_ts,
            cx=500.0, cy=500.0,
            joint_index_subset=[5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        # Joint 0 (subset[0]=5) -> X=1, u = 100*1/10+500 = 510
        # Joint 1 (subset[1]=6) -> X=-1, u = 100*-1/10+500 = 490
        np.testing.assert_allclose(out[0, 0, 0], [510.0, 500.0])
        np.testing.assert_allclose(out[0, 0, 1], [490.0, 500.0])


# ---------------------------------------------------------------------------
# scale_vitpose_to_native
# ---------------------------------------------------------------------------


class TestScaleVitposeToNative:
    def test_no_op_when_canvas_equals_native(self):
        vp = np.array([[100.0, 200.0, 0.9], [300.0, 400.0, 0.8]], dtype=np.float64)
        out = scale_vitpose_to_native(vp, phmr_canvas_wh=(1280, 720), native_wh=(1280, 720))
        np.testing.assert_allclose(out, vp)

    def test_portrait_upscale(self):
        vp = np.array([[252.0, 448.0, 0.9]], dtype=np.float64)
        out = scale_vitpose_to_native(vp, phmr_canvas_wh=(504, 896), native_wh=(576, 1024))
        # X: 252 * 576/504 = 288, Y: 448 * 1024/896 = 512
        np.testing.assert_allclose(out[0, 0], 288.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 1], 512.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 2], 0.9)

    def test_preserves_nan(self):
        vp = np.array([[np.nan, np.nan, 0.0]], dtype=np.float64)
        out = scale_vitpose_to_native(vp, phmr_canvas_wh=(504, 896), native_wh=(576, 1024))
        assert np.isnan(out[0, 0])
        assert np.isnan(out[0, 1])

    def test_preserves_higher_dim(self):
        vp = np.full((188, 5, 17, 3), 100.0, dtype=np.float64)
        out = scale_vitpose_to_native(vp, phmr_canvas_wh=(504, 896), native_wh=(576, 1024))
        assert out.shape == (188, 5, 17, 3)
        # X scaled by 576/504, Y by 1024/896
        np.testing.assert_allclose(out[..., 0], 100.0 * 576 / 504)
        np.testing.assert_allclose(out[..., 1], 100.0 * 1024 / 896)
        np.testing.assert_allclose(out[..., 2], 100.0)


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


def _make_synthetic_clip(tmp_path):
    """Build a tiny but real-shape clip directory layout."""
    import joblib
    clip = tmp_path / "synth"
    inter = clip / "intermediates" / "frames_full"
    inter.mkdir(parents=True)
    Image.new("RGB", (1280, 720), color=(0, 0, 0)).save(inter / "00000000.jpg")

    phmr = clip / "prompthmr"
    phmr.mkdir()
    # PHMR canvas == native; 2 dancers
    results = {
        "camera": {
            "img_focal": np.float64(1280.0),
            "img_center": [640.0, 360.0],
            "pred_cam_R": np.zeros((1, 3, 3), dtype=np.float32),
            "pred_cam_T": np.zeros((1, 3), dtype=np.float32),
        },
        "people": {
            1: {
                "frames": np.array([0]),
                # All 17 keypoints at (640, 360) with conf 0.9
                "vitpose": np.array([[[640.0, 360.0, 0.9]] * 17], dtype=np.float64),
            },
            2: {
                "frames": np.array([0]),
                "vitpose": np.array([[[640.0, 360.0, 0.9]] * 17], dtype=np.float64),
            },
        },
    }
    joblib.dump(results, phmr / "results.pkl")

    b4d = clip / "sam_body4d"
    (b4d / "focal_4d_individual" / "1").mkdir(parents=True)
    (b4d / "focal_4d_individual" / "2").mkdir(parents=True)
    # cam_t puts the joint at exactly the image center (X=Y=0, Z>0)
    (b4d / "focal_4d_individual" / "1" / "00000000.json").write_text(json.dumps(
        {"focal_length": 1280.0, "camera": [0.0, 0.0, 5.0]}
    ))
    (b4d / "focal_4d_individual" / "2" / "00000000.json").write_text(json.dumps(
        {"focal_length": 1280.0, "camera": [0.0, 0.0, 5.0]}
    ))
    # joints_world = (T=1, N=2, 70, 3); all zeros so cam-frame == cam_t == (0,0,5)
    j = np.zeros((1, 2, 70, 3), dtype=np.float32)
    np.save(b4d / "joints_world.npy", j)
    return clip


class TestMainEndToEnd:
    def test_writes_metrics_with_body4d_fields(self, tmp_path):
        clip = _make_synthetic_clip(tmp_path)
        out_path = clip / "comparison" / "reproj_metrics.json"
        rc = main([
            "--prompthmr-dir", str(clip / "prompthmr"),
            "--body4d-dir", str(clip / "sam_body4d"),
            "--frames-dir", str(clip / "intermediates" / "frames_full"),
            "--output", str(out_path),
            "--vitpose-conf-threshold", "0.3",
        ])
        assert rc == 0
        m = json.loads(out_path.read_text())
        # Tests the SCHEMA, not the exact value
        assert "mean_mpjpe_body4d_vs_vitpose_px" in m
        assert "per_joint_mpjpe_body4d_vs_vitpose_px" in m
        assert "body4d_native_image_w" in m
        assert "body4d_native_image_h" in m
        # All joints land at (640, 360) by construction; ViTPose also at (640, 360)
        # => pixel error should be ~0 (allow tiny float precision)
        assert m["mean_mpjpe_body4d_vs_vitpose_px"] < 1e-3, \
            f"expected ~0 px, got {m['mean_mpjpe_body4d_vs_vitpose_px']}"

    def test_extends_existing_reproj_file(self, tmp_path):
        clip = _make_synthetic_clip(tmp_path)
        out_path = clip / "comparison" / "reproj_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Pre-existing file from PHMR side
        out_path.write_text(json.dumps({
            "schema_version": 1,
            "mean_mpjpe_phmr_vs_vitpose_px": 9.99,
            "phmr_focal": 1280.0,
        }))
        rc = main([
            "--prompthmr-dir", str(clip / "prompthmr"),
            "--body4d-dir", str(clip / "sam_body4d"),
            "--frames-dir", str(clip / "intermediates" / "frames_full"),
            "--output", str(out_path),
        ])
        assert rc == 0
        m = json.loads(out_path.read_text())
        # Existing PHMR fields preserved
        assert m["mean_mpjpe_phmr_vs_vitpose_px"] == 9.99
        assert m["phmr_focal"] == 1280.0
        # New Body4D fields added
        assert "mean_mpjpe_body4d_vs_vitpose_px" in m
