"""Unit tests for the PromptHMR SAM-2 mask sidecar.

Only the GPU-free helper functions are unit-tested here. The full SAM-2
propagation path is exercised by the box-side smoke test in plan Task 6
step 2.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from threed.sidecar_promthmr.build_masks import (
    DAVIS_PALETTE,
    assemble_palette_canvas,
    chdir_to_prompthmr,
    compute_union,
    davis_palette,
    hydra_absolute_config_name,
    inject_prompthmr_path,
    load_video_frames_bgr,
    resolve_default_sam2_paths,
    valid_frames_set,
)


class TestDavisPalette:
    def test_returns_768_bytes(self):
        assert len(DAVIS_PALETTE) == 768

    def test_index_zero_is_black(self):
        assert DAVIS_PALETTE[0:3] == bytes([0, 0, 0])

    def test_index_one_is_not_black(self):
        assert DAVIS_PALETTE[3:6] != bytes([0, 0, 0])

    def test_palette_is_deterministic(self):
        assert davis_palette() == davis_palette()


class TestResolveDefaultSam2Paths:
    def test_returns_tiny_ckpt_inside_prompthmr_tree(self, tmp_path):
        ckpt, cfg = resolve_default_sam2_paths(tmp_path)
        assert ckpt == tmp_path / "data" / "pretrain" / "sam2_ckpts" / "sam2_hiera_tiny.pt"
        assert cfg == "pipeline/sam2/sam2_hiera_t.yaml"


class TestValidFramesSet:
    def test_two_tracks(self):
        tracks = {
            1: {"frames": np.array([0, 1, 2, 5], dtype=np.int64)},
            2: {"frames": np.array([3, 4], dtype=np.int64)},
        }
        result = valid_frames_set(tracks)
        assert result == {1: {0, 1, 2, 5}, 2: {3, 4}}

    def test_string_keys_get_coerced_to_int(self):
        # joblib sometimes round-trips int keys as numpy int
        tracks = {np.int64(7): {"frames": np.array([10, 11], dtype=np.int64)}}
        result = valid_frames_set(tracks)
        assert result == {7: {10, 11}}

    def test_empty_track(self):
        tracks = {7: {"frames": np.array([], dtype=np.int64)}}
        assert valid_frames_set(tracks) == {7: set()}


class TestAssemblePaletteCanvas:
    def test_single_tid_writes_uniform_value(self):
        H, W = 4, 4
        msk = np.zeros((H, W), dtype=bool)
        msk[1:3, 1:3] = True
        result = assemble_palette_canvas({3: msk}, H, W)
        assert result.dtype == np.uint8
        assert result.shape == (H, W)
        assert result[1, 1] == 3
        assert result[1, 2] == 3
        assert result[0, 0] == 0
        assert result[3, 3] == 0

    def test_overlap_larger_tid_wins(self):
        H, W = 4, 4
        msk_small_tid = np.ones((H, W), dtype=bool)
        msk_big_tid = np.zeros((H, W), dtype=bool)
        msk_big_tid[1:3, 1:3] = True
        result = assemble_palette_canvas({1: msk_small_tid, 5: msk_big_tid}, H, W)
        assert result[1, 1] == 5  # overlap region: tid 5 wins
        assert result[0, 0] == 1  # no overlap: tid 1 stays
        assert result[3, 3] == 1

    def test_empty_dict_gives_zero_canvas(self):
        result = assemble_palette_canvas({}, 3, 3)
        assert result.dtype == np.uint8
        assert (result == 0).all()


class TestComputeUnion:
    def test_two_frames_two_tids(self):
        H, W = 3, 3
        m_f0_t1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
        m_f0_t2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
        m_f1_t1 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
        per_frame_per_tid = {
            0: {1: m_f0_t1, 2: m_f0_t2},
            1: {1: m_f1_t1},
        }
        union = compute_union(per_frame_per_tid, n_frames=2, H=H, W=W)
        assert union.shape == (2, 3, 3)
        assert union.dtype == bool
        assert union[0, 0, 0]
        assert union[0, 0, 1]
        assert not union[0, 0, 2]
        assert union[1, 1, 0]
        assert not union[1, 0, 0]

    def test_no_frames_with_data(self):
        union = compute_union({}, n_frames=5, H=2, W=2)
        assert union.shape == (5, 2, 2)
        assert union.sum() == 0


class TestInjectPromptHmrPath:
    def test_path_appears_at_front_of_sys_path(self, tmp_path, monkeypatch):
        import sys
        original = list(sys.path)
        try:
            inject_prompthmr_path(tmp_path)
            assert sys.path[0] == str(tmp_path.resolve())
        finally:
            sys.path[:] = original

    def test_double_inject_does_not_duplicate(self, tmp_path):
        import sys
        original = list(sys.path)
        try:
            inject_prompthmr_path(tmp_path)
            first = list(sys.path)
            inject_prompthmr_path(tmp_path)
            second = list(sys.path)
            # The path should appear exactly once across both inserts
            assert second.count(str(tmp_path.resolve())) == 1
            assert first == second
        finally:
            sys.path[:] = original


class TestChdirToPromptHmr:
    def test_changes_cwd_and_returns_previous(self, tmp_path):
        import os
        original = Path.cwd()
        try:
            previous = chdir_to_prompthmr(tmp_path)
            assert previous == original
            assert Path.cwd() == tmp_path.resolve()
        finally:
            os.chdir(original)

    def test_chdir_required_for_pipeline_gvhmr_relative_path(self, tmp_path):
        """Regression for the 2026-04-18 ModuleNotFoundError: no module 'hmr4d'.

        PromptHMR's ``pipeline.phmr_vid`` does
        ``sys.path.insert(0, 'pipeline/gvhmr')``. That relative path only
        resolves when cwd is the PromptHMR root, so this test pins the
        contract that ``chdir_to_prompthmr`` makes ``Path('pipeline/gvhmr')``
        resolve under the given root.
        """
        import os
        original = Path.cwd()
        # Build a fake PromptHMR layout
        (tmp_path / "pipeline" / "gvhmr").mkdir(parents=True)
        try:
            chdir_to_prompthmr(tmp_path)
            relative = Path("pipeline/gvhmr").resolve()
            assert relative == (tmp_path / "pipeline" / "gvhmr").resolve()
        finally:
            os.chdir(original)


class TestHydraAbsoluteConfigName:
    def test_double_slash_prefix_with_absolute_filesystem_path(self, tmp_path):
        """Regression for the 2026-04-18 hydra MissingConfigException.

        The leading ``/`` plus an absolute filesystem path is the literal
        marker Hydra uses to bypass its search path. Without it Hydra
        would look up ``pipeline/sam2/sam2_hiera_t.yaml`` in
        ``pkg://sam2`` (the upstream sam2 package) and miss PromptHMR's
        local override entirely.
        """
        result = hydra_absolute_config_name(tmp_path, "pipeline/sam2/sam2_hiera_t.yaml")
        assert result.startswith("//")  # leading '/' + abspath that itself starts with '/'
        assert result.endswith("/pipeline/sam2/sam2_hiera_t.yaml")
        # The substring after the leading '/' must round-trip as a real abs path
        assert result[1:].startswith(str(tmp_path.resolve()))


class TestLoadVideoFramesBgr:
    """Regression for the 2026-04-18 ``init_state`` TypeError.

    PromptHMR's modified ``SAM2VideoPredictor.init_state`` requires a
    stacked ``video_frames`` array (it does ``video_frames.shape[1:3]``)
    instead of the upstream ``video_path``. The in-class
    ``_load_img_as_tensor`` accepts BGR numpy arrays without channel-swap,
    so the loader must emit ``(N, H, W, 3)`` BGR uint8.
    """

    def _make_frame(self, h: int, w: int, fill: int) -> np.ndarray:
        img = np.full((h, w, 3), fill, dtype=np.uint8)
        return img

    def test_returns_4d_uint8_in_sorted_order(self, tmp_path):
        import cv2

        for idx, fill in enumerate([10, 20, 30]):
            cv2.imwrite(str(tmp_path / f"{idx:05d}.jpg"), self._make_frame(8, 16, fill))
        result = load_video_frames_bgr(tmp_path)
        assert result.shape == (3, 8, 16, 3)
        assert result.dtype == np.uint8
        # JPEG is lossy, so check median (not exact equality) per frame
        assert abs(int(np.median(result[0])) - 10) <= 2
        assert abs(int(np.median(result[1])) - 20) <= 2
        assert abs(int(np.median(result[2])) - 30) <= 2

    def test_init_state_compatible_shape_unpacks_to_h_w(self, tmp_path):
        """``init_state`` does ``video_frames.shape[1:3]``; pin that contract."""
        import cv2

        cv2.imwrite(str(tmp_path / "00000.jpg"), self._make_frame(11, 17, 50))
        result = load_video_frames_bgr(tmp_path)
        h, w = result.shape[1:3]
        assert (h, w) == (11, 17)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_video_frames_bgr(tmp_path / "no_such_subdir")

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_video_frames_bgr(tmp_path)
