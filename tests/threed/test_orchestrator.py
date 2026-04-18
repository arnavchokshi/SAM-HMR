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
        assert "--max-frames" not in cmd  # default unset

    def test_max_frames_forwarded_when_set(self, orch, fake_dirs, tmp_path):
        cmd = orch.build_stage_a_cmd(
            python=Path("/usr/bin/python"),
            clip="loveTest",
            video=Path("/tmp/foo.mov"),
            cache_dir=tmp_path / "cache",
            max_frames=188,
        )
        assert "--max-frames" in cmd
        assert "188" in cmd

    def test_max_frames_none_means_no_flag(self, orch, fake_dirs, tmp_path):
        cmd = orch.build_stage_a_cmd(
            python=Path("/usr/bin/python"),
            clip="loveTest",
            video=Path("/tmp/foo.mov"),
            cache_dir=tmp_path / "cache",
            max_frames=None,
        )
        assert "--max-frames" not in cmd

    def test_output_root_forwarded_when_set(self, orch, fake_dirs, tmp_path):
        cmd = orch.build_stage_a_cmd(
            python=Path("/usr/bin/python"),
            clip="loveTest",
            video=Path("/tmp/foo.mov"),
            cache_dir=tmp_path / "cache",
            output_root=Path("/home/ubuntu/work/3d_compare"),
        )
        assert "--output-root" in cmd
        assert "/home/ubuntu/work/3d_compare" in cmd

    def test_output_root_none_means_no_flag(self, orch, fake_dirs, tmp_path):
        cmd = orch.build_stage_a_cmd(
            python=Path("/usr/bin/python"),
            clip="loveTest",
            video=Path("/tmp/foo.mov"),
            cache_dir=tmp_path / "cache",
            output_root=None,
        )
        assert "--output-root" not in cmd


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

    def test_render_overlay(self, orch, fake_dirs):
        """Stage C1.5 — per-frame SMPL-X overlay JPGs (closes Stage D followup #3)."""
        cmd = orch.build_phmr_render_overlay_cmd(
            prompthmr_dir=fake_dirs.prompthmr,
            frames_dir=fake_dirs.intermediates / "frames",
            smplx_path=Path("/x/PromptHMR/data/body_models/smplx"),
        )
        assert "threed.sidecar_promthmr.render_overlay" in cmd
        assert "--prompthmr-dir" in cmd
        assert "--frames-dir" in cmd
        assert "--smplx-path" in cmd
        assert "/x/PromptHMR/data/body_models/smplx" in cmd


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

    def test_render_overlay(self, orch, fake_dirs):
        """Stage C2.5 — per-frame body4d overlay JPGs (matches PHMR overlay)."""
        cmd = orch.build_body4d_render_overlay_cmd(
            body4d_dir=fake_dirs.sam_body4d,
            frames_dir=fake_dirs.intermediates / "frames_full",
        )
        assert "threed.sidecar_body4d.render_overlay" in cmd
        assert "--body4d-dir" in cmd
        assert "--frames-dir" in cmd
        assert str(fake_dirs.intermediates / "frames_full") in cmd


class TestCompareCmd:
    def test_run_compare(self, orch, fake_dirs):
        cmd = orch.build_compare_cmd(
            prompthmr_joints=fake_dirs.prompthmr / "joints_coco17_cam.npy",
            body4d_joints=fake_dirs.sam_body4d / "joints_world.npy",
            output=fake_dirs.comparison / "metrics.json",
        )
        assert "threed.compare.run_compare" in cmd

    def test_run_compare_forwards_world_joints(self, orch, fake_dirs):
        """Followup #2: orchestrator wires PHMR's joints_world.npy through."""
        cmd = orch.build_compare_cmd(
            prompthmr_joints=fake_dirs.prompthmr / "joints_coco17_cam.npy",
            body4d_joints=fake_dirs.sam_body4d / "joints_world.npy",
            output=fake_dirs.comparison / "metrics.json",
            prompthmr_world_joints=fake_dirs.prompthmr / "joints_world.npy",
        )
        assert "--prompthmr-world-joints" in cmd
        assert str(fake_dirs.prompthmr / "joints_world.npy") in cmd

    def test_run_compare_omits_world_joints_when_none(self, orch, fake_dirs):
        cmd = orch.build_compare_cmd(
            prompthmr_joints=fake_dirs.prompthmr / "joints_coco17_cam.npy",
            body4d_joints=fake_dirs.sam_body4d / "joints_world.npy",
            output=fake_dirs.comparison / "metrics.json",
            prompthmr_world_joints=None,
        )
        assert "--prompthmr-world-joints" not in cmd

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
            skip_stage_a=False, skip_phmr=False, skip_body4d=False,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert plan == ["stage_a", "phmr_masks", "phmr_run", "phmr_project",
                        "phmr_render", "body4d", "body4d_render",
                        "compare", "render"]

    def test_skip_stage_a(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=True, skip_phmr=False, skip_body4d=False,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert "stage_a" not in plan

    def test_skip_phmr(self, orch):
        """--skip-phmr drops every PHMR stage (masks/run/project/render)."""
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=True, skip_body4d=False,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert not any(s.startswith("phmr") for s in plan)

    def test_skip_phmr_render_only(self, orch):
        """--skip-phmr-render keeps the rest of the PHMR chain but drops the overlay."""
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=False,
            skip_compare=False, skip_phmr_render=True, skip_body4d_render=False,
        )
        assert "phmr_render" not in plan
        assert "phmr_run" in plan and "phmr_project" in plan

    def test_skip_phmr_implies_skip_phmr_render(self, orch):
        """If the whole PHMR chain is skipped, the render flag is moot."""
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=True, skip_body4d=False,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert "phmr_render" not in plan

    def test_skip_body4d(self, orch):
        """--skip-body4d drops both body4d and body4d_render — can't render
        meshes that were never produced."""
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=True,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert "body4d" not in plan
        assert "body4d_render" not in plan

    def test_skip_body4d_render_only(self, orch):
        """--skip-body4d-render keeps body4d itself but drops the overlay
        render. Useful when iterating on Stage D — metrics still update,
        the side-by-side falls back to upstream's clean-bg ``rendered_frames/``."""
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=False,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=True,
        )
        assert "body4d" in plan
        assert "body4d_render" not in plan

    def test_skip_body4d_implies_skip_body4d_render(self, orch):
        """If body4d itself is skipped, body4d_render is dropped too."""
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=True,
            skip_compare=False, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert "body4d_render" not in plan

    def test_skip_compare(self, orch):
        plan = orch.plan_pipeline(
            skip_stage_a=False, skip_phmr=False, skip_body4d=False,
            skip_compare=True, skip_phmr_render=False, skip_body4d_render=False,
        )
        assert "compare" not in plan and "render" not in plan
