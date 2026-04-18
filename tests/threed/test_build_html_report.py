"""Unit tests for :mod:`scripts.build_html_report` (Followup #5)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.build_html_report import (
    aggregate_clip_metrics,
    build_html,
    discover_clips,
    main,
    summarize_clip,
)


# ---------------------------------------------------------------------------
# Synthetic clip layout fixture
# ---------------------------------------------------------------------------


def _write_metrics(clip_dir: Path, *, mpjpe_pa_mean: float, foot_skating_world: float):
    """Write a minimal metrics.json that mirrors the real Stage D output."""
    metrics = {
        "schema_version": 1,
        "n_frames_compared": 188,
        "n_dancers_compared": 3,
        "n_frames_phmr": 188,
        "n_frames_body4d": 188,
        "n_dancers_phmr": 3,
        "n_dancers_body4d": 3,
        "raw_joint_axis_phmr": 17,
        "raw_joint_axis_body4d": 70,
        "per_joint_jitter_phmr_m_per_frame": np.full((3, 17), 0.04).tolist(),
        "per_joint_jitter_body4d_m_per_frame": np.full((3, 17), 0.05).tolist(),
        "per_joint_mpjpe_m": np.full((3, 17), 5.0).tolist(),
        "per_joint_mpjpe_pa_m": np.full((3, 17), mpjpe_pa_mean).tolist(),
        "foot_skating_phmr_m_per_frame": [0.0, 0.0, 0.0],
        "foot_skating_body4d_m_per_frame": [0.05, 0.04, 0.06],
        "foot_skating_phmr_world_m_per_frame": [foot_skating_world] * 3,
        "joint_layout": "COCO-17",
        "joint_names": [
            "nose","leye","reye","lear","rear",
            "lshoulder","rshoulder","lelbow","relbow","lwrist","rwrist",
            "lhip","rhip","lknee","rknee","lankle","rankle",
        ],
        "foot_idx": 15,
        "foot_threshold_m": 0.05,
    }
    (clip_dir / "comparison").mkdir(parents=True, exist_ok=True)
    (clip_dir / "comparison" / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _write_reproj(clip_dir: Path, *, mean_px: float):
    """Write reproj_metrics.json that mirrors the Followup #4 output."""
    reproj = {
        "schema_version": 1,
        "joint_layout": "COCO-17",
        "n_frames": 188,
        "n_dancers": 3,
        "vitpose_conf_threshold": 0.3,
        "n_low_confidence_keypoints": 50,
        "phmr_focal": 1280.0,
        "phmr_cx": 640.0,
        "phmr_cy": 360.0,
        "tids_sorted": [1, 2, 3],
        "valid_frames_per_dancer_phmr": [188, 188, 188],
        "per_joint_mpjpe_phmr_vs_vitpose_px": np.full((3, 17), mean_px).tolist(),
        "mean_mpjpe_phmr_vs_vitpose_px": mean_px,
    }
    (clip_dir / "comparison" / "reproj_metrics.json").write_text(json.dumps(reproj, indent=2))


def _make_video(p: Path):
    """Touch a non-empty placeholder MP4 (the report only links to it)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverClips:
    def test_finds_clips_with_metrics_json(self, tmp_path: Path):
        for clip in ["alpha", "beta"]:
            _write_metrics(tmp_path / clip, mpjpe_pa_mean=0.3, foot_skating_world=0.01)
        clips = discover_clips(tmp_path)
        assert sorted(c.name for c in clips) == ["alpha", "beta"]

    def test_skips_dirs_without_metrics_json(self, tmp_path: Path):
        (tmp_path / "no_metrics").mkdir()
        _write_metrics(tmp_path / "good", mpjpe_pa_mean=0.3, foot_skating_world=0.01)
        clips = discover_clips(tmp_path)
        assert [c.name for c in clips] == ["good"]

    def test_returns_sorted(self, tmp_path: Path):
        for clip in ["zebra", "apple", "mango"]:
            _write_metrics(tmp_path / clip, mpjpe_pa_mean=0.3, foot_skating_world=0.01)
        clips = discover_clips(tmp_path)
        assert [c.name for c in clips] == ["apple", "mango", "zebra"]


class TestSummarizeClip:
    def test_summary_dict_has_expected_fields(self, tmp_path: Path):
        clip = tmp_path / "demo"
        _write_metrics(clip, mpjpe_pa_mean=0.42, foot_skating_world=0.02)
        _write_reproj(clip, mean_px=12.34)
        s = summarize_clip(clip)
        assert s["name"] == "demo"
        assert s["n_frames"] == 188
        assert s["n_dancers"] == 3
        np.testing.assert_allclose(s["mean_mpjpe_pa_m"], 0.42, atol=1e-6)
        np.testing.assert_allclose(s["mean_jitter_phmr_m"], 0.04, atol=1e-6)
        np.testing.assert_allclose(s["mean_jitter_body4d_m"], 0.05, atol=1e-6)
        np.testing.assert_allclose(s["mean_foot_skating_phmr_world_m"], 0.02, atol=1e-6)
        np.testing.assert_allclose(s["mean_reproj_mpjpe_phmr_px"], 12.34, atol=1e-6)
        assert "side_by_side_path" in s

    def test_handles_missing_reproj_metrics(self, tmp_path: Path):
        """A clip with no reproj_metrics.json should still summarise (None placeholders)."""
        clip = tmp_path / "demo"
        _write_metrics(clip, mpjpe_pa_mean=0.42, foot_skating_world=0.02)
        s = summarize_clip(clip)
        assert s["mean_reproj_mpjpe_phmr_px"] is None

    def test_handles_missing_world_foot_skating(self, tmp_path: Path):
        """Older clips without world FS should yield None, not crash."""
        clip = tmp_path / "demo"
        _write_metrics(clip, mpjpe_pa_mean=0.42, foot_skating_world=0.0)
        m = json.loads((clip / "comparison" / "metrics.json").read_text())
        m.pop("foot_skating_phmr_world_m_per_frame")
        (clip / "comparison" / "metrics.json").write_text(json.dumps(m))
        s = summarize_clip(clip)
        assert s["mean_foot_skating_phmr_world_m"] is None


class TestAggregateClipMetrics:
    def test_aggregate_returns_one_row_per_clip(self, tmp_path: Path):
        for c, pa in [("alpha", 0.10), ("beta", 0.20), ("gamma", 0.30)]:
            _write_metrics(tmp_path / c, mpjpe_pa_mean=pa, foot_skating_world=0.01)
            _write_reproj(tmp_path / c, mean_px=10.0)
        rows = aggregate_clip_metrics(tmp_path)
        assert [r["name"] for r in rows] == ["alpha", "beta", "gamma"]
        np.testing.assert_allclose(
            [r["mean_mpjpe_pa_m"] for r in rows], [0.10, 0.20, 0.30],
        )


class TestBuildHtml:
    def test_html_contains_all_clip_names_and_video_paths(self, tmp_path: Path):
        for c in ["alpha", "beta"]:
            _write_metrics(tmp_path / c, mpjpe_pa_mean=0.3, foot_skating_world=0.01)
            _write_reproj(tmp_path / c, mean_px=12.3)
            _make_video(tmp_path / c / "comparison" / "side_by_side.mp4")
        rows = aggregate_clip_metrics(tmp_path)
        html = build_html(rows, root=tmp_path, title="Test Report")
        assert "<html" in html
        assert "Test Report" in html
        for c in ["alpha", "beta"]:
            assert c in html
            assert f"{c}/comparison/side_by_side.mp4" in html
            assert f'<source src="{c}/comparison/side_by_side.mp4"' in html
        assert "12.30" in html

    def test_html_table_includes_summary_columns(self, tmp_path: Path):
        _write_metrics(tmp_path / "alpha", mpjpe_pa_mean=0.42, foot_skating_world=0.025)
        _write_reproj(tmp_path / "alpha", mean_px=11.0)
        rows = aggregate_clip_metrics(tmp_path)
        html = build_html(rows, root=tmp_path, title="t")
        for col in ["MPJPE", "PA-MPJPE", "Jitter", "Foot-skating", "Reproj"]:
            assert col in html

    def test_handles_clip_with_no_video(self, tmp_path: Path):
        """If side_by_side.mp4 is missing the report should still render with a placeholder."""
        _write_metrics(tmp_path / "alpha", mpjpe_pa_mean=0.42, foot_skating_world=0.025)
        rows = aggregate_clip_metrics(tmp_path)
        html = build_html(rows, root=tmp_path, title="t")
        assert "alpha" in html
        assert "no video" in html.lower() or "missing" in html.lower()


class TestMainEndToEnd:
    def test_writes_report_html_and_no_op_smoke(self, tmp_path: Path):
        for c in ["alpha", "beta"]:
            _write_metrics(tmp_path / c, mpjpe_pa_mean=0.3, foot_skating_world=0.01)
            _write_reproj(tmp_path / c, mean_px=10.0)
            _make_video(tmp_path / c / "comparison" / "side_by_side.mp4")
        out = tmp_path / "report.html"
        rc = main([
            "--root", str(tmp_path),
            "--output", str(out),
            "--title", "Smoke Report",
        ])
        assert rc == 0
        assert out.is_file()
        html = out.read_text()
        assert "Smoke Report" in html
        for c in ["alpha", "beta"]:
            assert c in html
        # The HTML must be valid-enough to mention each video relative path
        assert re.search(r'<source\s+src="alpha/comparison/side_by_side\.mp4"', html)
