"""Unit tests for ``scripts/run_3d_compare.py`` (plan Task 12).

The orchestrator shells out to multi-env subprocess invocations, so we
test the pieces that have logic — the per-stage command builders + the
skip-flag handling — without ever touching subprocess. The real
end-to-end smoke runs on the box (the orchestrator's main() is only
defensible glue around well-tested per-stage modules).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_orchestrator():
    """Import scripts/run_3d_compare.py without it being on sys.path."""
    here = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_3d_compare", here / "scripts" / "run_3d_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_3d_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def orch():
    return _load_orchestrator()


@pytest.fixture
def fake_dirs(tmp_path: Path):
    """Build a minimal ClipDirs-like object."""
    class Dirs:
        intermediates = tmp_path / "intermediates"
        prompthmr = tmp_path / "prompthmr"
        sam_body4d = tmp_path / "sam_body4d"
        comparison = tmp_path / "comparison"
    for d in (Dirs.intermediates, Dirs.prompthmr, Dirs.sam_body4d, Dirs.comparison):
        d.mkdir(parents=True, exist_ok=True)
    return Dirs


class TestStageACmd:
    def test_basic(self, orch, fake_dirs, tmp_path):
        cmd = orch.build_stage_a_cmd(
            python=Path("/usr/bin/python"),
            clip="adiTest",
            video=Path("/tmp/foo.mov"),
            cache_dir=tmp_path / "cache",
        )
        assert cmd[0] == "/usr/bin/python"
        assert "threed.stage_a.run_stage_a" in cmd
        assert "--clip" in cmd and "adiTest" in cmd
        assert "--video" in cmd and "/tmp/foo.mov" in cmd
        assert "--cache-dir" in cmd


class TestPromptHmrCmds:
    def test_build_masks(self, orch, fake_dirs):
        cmd = orch.build_phmr_masks_cmd(
            intermediates_dir=fake_dirs.intermediates,
            prompthmr_path=Path("/x/PromptHMR"),
        )
        assert "threed.sidecar_promthmr.build_masks" in cmd
        assert "--intermediates-dir" in cmd
        assert "--prompthmr-path" in cmd
        assert "/x/PromptHMR" in cmd

    def test_run_phmr_default(self, orch, fake_dirs):
        cmd = orch.build_phmr_run_cmd(
            intermediates_dir=fake_dirs.intermediates,
            output_dir=fake_dirs.prompthmr,
            prompthmr_path=Path("/x/PromptHMR"),
            static_camera=False,
        )
        assert "threed.sidecar_promthmr.run_promthmr_vid" in cmd
        assert "--static-camera" not in cmd

    def test_run_phmr_static_cam(self, orch, fake_dirs):
        cmd = orch.build_phmr_run_cmd(
            intermediates_dir=fake_dirs.intermediates,
            output_dir=fake_dirs.prompthmr,
            prompthmr_path=Path("/x/PromptHMR"),
            static_camera=True,
        )
        assert "--static-camera" in cmd

    def test_project_joints(self, orch, fake_dirs):
        cmd = orch.build_phmr_project_joints_cmd(
            prompthmr_dir=fake_dirs.prompthmr,
        )
        assert "threed.sidecar_promthmr.project_joints" in cmd
        assert "--prompthmr-dir" in cmd
        assert "joints_coco17_cam.npy" in " ".join(cmd)


class TestBody4dCmd:
    def test_default(self, orch, fake_dirs):
        cmd = orch.build_body4d_cmd(
            intermediates_dir=fake_dirs.intermediates,
            output_dir=fake_dirs.sam_body4d,
            sam_body4d_path=Path("/x/sam-body4d"),
            disable_completion=False,
            batch_size=None,
        )
        assert "threed.sidecar_body4d.run_body4d" in cmd
        assert "--disable-completion" not in cmd
        assert "--batch-size" not in cmd

    def test_with_overrides(self, orch, fake_dirs):
        cmd = orch.build_body4d_cmd(
            intermediates_dir=fake_dirs.intermediates,
            output_dir=fake_dirs.sam_body4d,
            sam_body4d_path=Path("/x/sam-body4d"),
            disable_completion=True,
            batch_size=32,
        )
        assert "--disable-completion" in cmd
        assert "--batch-size" in cmd
        assert "32" in cmd


class TestCompareCmd:
    def test_run_compare(self, orch, fake_dirs):
        cmd = orch.build_compare_cmd(
            prompthmr_joints=fake_dirs.prompthmr / "joints_coco17_cam.npy",
            body4d_joints=fake_dirs.sam_body4d / "joints_world.npy",
            output=fake_dirs.comparison / "metrics.json",
        )
        assert "threed.compare.run_compare" in cmd

    def test_render(self, orch, fake_dirs):
        cmd = orch.build_render_cmd(
            body4d_frames_dir=fake_dirs.sam_body4d / "rendered_frames",
            prompthmr_frames_dir=None,
            output=fake_dirs.comparison / "side_by_side.mp4",
            fps=30,
        )
        assert "threed.compare.render" in cmd
        assert "--fps" in cmd and "30" in cmd
        assert "--prompthmr-frames-dir" not in cmd

    def test_render_with_phmr_frames(self, orch, fake_dirs):
        cmd = orch.build_render_cmd(
            body4d_frames_dir=fake_dirs.sam_body4d / "rendered_frames",
            prompthmr_frames_dir=fake_dirs.prompthmr / "phmr_frames",
            output=fake_dirs.comparison / "side_by_side.mp4",
            fps=30,
        )
        assert "--prompthmr-frames-dir" in cmd


class TestPipelinePlan:
    """Verify the high-level plan respects --skip flags without invoking subprocesses."""

    def test_all_stages_default(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=False, skip_compare=False
        )
        assert plan == ["stage_a", "phmr_masks", "phmr_run", "phmr_project",
                        "body4d", "compare", "render"]

    def test_skip_stage_a(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=True, skip_phmr=False, skip_body4d=False, skip_compare=False
        )
        assert "stage_a" not in plan

    def test_skip_phmr(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=True, skip_body4d=False, skip_compare=False
        )
        assert not any(s.startswith("phmr") for s in plan)

    def test_skip_body4d(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=True, skip_compare=False
        )
        assert "body4d" not in plan

    def test_skip_compare(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=False, skip_compare=True
        )
        assert "compare" not in plan and "render" not in plan
