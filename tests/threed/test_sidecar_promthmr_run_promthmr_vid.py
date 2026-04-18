"""Unit tests for the PromptHMR-Vid sidecar runner.

GPU-free coverage of the helper functions. The end-to-end PromptHMR-Vid
inference path is exercised by the box-side smoke test in plan Task 7
step 2 (CUDA-only) since DROID-SLAM, ViTPose, and PromptHMR-Vid all
require CUDA wheels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.sidecar_promthmr.run_promthmr_vid import (
    intermediates_layout_ok,
    joints_world_padded,
    load_per_track_masks,
    sorted_tid_list,
)


class TestIntermediatesLayoutOk:
    """Pin the exact files we require so error messages are clear."""

    def _make_layout(self, root: Path) -> None:
        (root / "frames").mkdir(parents=True)
        (root / "frames_full").mkdir()
        (root / "masks_per_track").mkdir()
        (root / "masks_palette").mkdir()
        (root / "tracks.pkl").write_bytes(b"")
        (root / "masks_union.npy").write_bytes(b"")
        for i in range(2):
            cv2.imwrite(str(root / "frames" / f"{i:08d}.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))

    def test_complete_layout_returns_no_errors(self, tmp_path):
        self._make_layout(tmp_path)
        ok, errs = intermediates_layout_ok(tmp_path)
        assert ok and errs == []

    def test_missing_frames_dir_reported(self, tmp_path):
        self._make_layout(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "frames")
        ok, errs = intermediates_layout_ok(tmp_path)
        assert not ok and any("frames" in e for e in errs)

    def test_missing_tracks_pkl_reported(self, tmp_path):
        self._make_layout(tmp_path)
        (tmp_path / "tracks.pkl").unlink()
        ok, errs = intermediates_layout_ok(tmp_path)
        assert not ok and any("tracks.pkl" in e for e in errs)

    def test_missing_masks_union_reported(self, tmp_path):
        self._make_layout(tmp_path)
        (tmp_path / "masks_union.npy").unlink()
        ok, errs = intermediates_layout_ok(tmp_path)
        assert not ok and any("masks_union.npy" in e for e in errs)

    def test_missing_masks_per_track_reported(self, tmp_path):
        self._make_layout(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "masks_per_track")
        ok, errs = intermediates_layout_ok(tmp_path)
        assert not ok and any("masks_per_track" in e for e in errs)

    def test_empty_frames_dir_reported(self, tmp_path):
        self._make_layout(tmp_path)
        for p in (tmp_path / "frames").iterdir():
            p.unlink()
        ok, errs = intermediates_layout_ok(tmp_path)
        assert not ok and any("frames" in e for e in errs)


class TestLoadPerTrackMasks:
    """Pin the dict-mutation contract Pipeline.results['people'] expects."""

    def _write_per_tid(self, root: Path, tid: int, frames: list[int], H: int, W: int) -> None:
        d = root / "masks_per_track" / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        for f in frames:
            m = np.zeros((H, W), dtype=np.uint8)
            m[1:3, 1:3] = 255
            cv2.imwrite(str(d / f"{f:08d}.png"), m)

    def test_populates_masks_track_id_detected_for_each_track(self, tmp_path):
        H, W = 4, 6
        tracks = {
            1: {"frames": np.array([0, 1, 2], dtype=np.int64), "bboxes": np.zeros((3, 4))},
            2: {"frames": np.array([1, 3], dtype=np.int64), "bboxes": np.zeros((2, 4))},
        }
        self._write_per_tid(tmp_path, 1, [0, 1, 2], H, W)
        self._write_per_tid(tmp_path, 2, [1, 3], H, W)
        result = load_per_track_masks(tmp_path, tracks, H=H, W=W)

        # Returned dict is the same object we passed in (mutates in place by design).
        assert result is tracks
        assert tracks[1]["track_id"] == 1
        assert tracks[2]["track_id"] == 2
        assert tracks[1]["masks"].shape == (3, H, W)
        assert tracks[2]["masks"].shape == (2, H, W)
        assert tracks[1]["masks"].dtype == bool
        assert tracks[2]["masks"].dtype == bool
        # Detected vector is all True (we trust DeepOcSort's per-frame detection).
        assert tracks[1]["detected"].tolist() == [True, True, True]
        assert tracks[2]["detected"].tolist() == [True, True]

    def test_missing_png_falls_back_to_zero_mask(self, tmp_path):
        H, W = 3, 4
        tracks = {7: {"frames": np.array([0, 1], dtype=np.int64)}}
        self._write_per_tid(tmp_path, 7, [0], H, W)  # frame 1 PNG intentionally missing
        result = load_per_track_masks(tmp_path, tracks, H=H, W=W)
        assert result[7]["masks"][1].sum() == 0

    def test_track_id_is_python_int_not_numpy(self, tmp_path):
        """``Pipeline`` round-trips track_id through ``int(...)``; np.int64 keys break it."""
        H, W = 4, 4
        tracks = {np.int64(3): {"frames": np.array([0], dtype=np.int64)}}
        self._write_per_tid(tmp_path, 3, [0], H, W)
        result = load_per_track_masks(tmp_path, tracks, H=H, W=W)
        # The injected track_id is a python int (not numpy)
        v = list(result.values())[0]
        assert isinstance(v["track_id"], int)
        assert v["track_id"] == 3


class TestSortedTidList:
    def test_sorted_ints(self):
        tracks = {3: {}, 1: {}, 5: {}, 2: {}}
        assert sorted_tid_list(tracks) == [1, 2, 3, 5]

    def test_handles_numpy_int_keys(self):
        tracks = {np.int64(3): {}, np.int32(1): {}}
        assert sorted_tid_list(tracks) == [1, 3]

    def test_empty(self):
        assert sorted_tid_list({}) == []


class TestJointsWorldPadded:
    """Pin the comparison artifact shape: (n_frames, n_dancers, 22, 3) NaN-padded."""

    def test_full_coverage_two_dancers(self):
        per_track_joints = {
            1: (np.array([0, 1, 2]), np.full((3, 22, 3), 1.0, dtype=np.float32)),
            2: (np.array([0, 2]), np.full((2, 22, 3), 2.0, dtype=np.float32)),
        }
        out = joints_world_padded(per_track_joints, n_frames=3, tid_order=[1, 2])
        assert out.shape == (3, 2, 22, 3)
        assert out.dtype == np.float32
        # tid 1, frame 0 → 1.0
        assert (out[0, 0] == 1.0).all()
        # tid 2, frame 1 → NaN
        assert np.isnan(out[1, 1]).all()
        # tid 2, frame 2 → 2.0
        assert (out[2, 1] == 2.0).all()

    def test_no_dancers_yields_empty_dim(self):
        out = joints_world_padded({}, n_frames=4, tid_order=[])
        assert out.shape == (4, 0, 22, 3)

    def test_dancer_with_no_frames_all_nan(self):
        per_track_joints = {7: (np.array([], dtype=np.int64), np.zeros((0, 22, 3), dtype=np.float32))}
        out = joints_world_padded(per_track_joints, n_frames=3, tid_order=[7])
        assert out.shape == (3, 1, 22, 3)
        assert np.isnan(out).all()
