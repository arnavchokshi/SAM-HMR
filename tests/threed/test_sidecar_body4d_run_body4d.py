"""Unit tests for :mod:`threed.sidecar_body4d.run_body4d` GPU-free helpers (Task 10).

Only ``_inject_sam_body4d_path`` and ``_patch_config`` are unit-tested;
the rest of the runner is GPU-bound (instantiates SAM-3D-Body) and is
covered by the box-side smoke ``~/work/run_task10_smoke.sh``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from threed.sidecar_body4d import run_body4d as runner


class TestInjectSamBody4dPath:
    def test_appends_all_subdirs_in_order(self, tmp_path: Path, monkeypatch):
        repo = tmp_path / "sam-body4d"
        (repo / "scripts").mkdir(parents=True)
        (repo / "models" / "sam_3d_body").mkdir(parents=True)
        (repo / "models" / "diffusion_vas").mkdir(parents=True)
        original_sys_path = list(sys.path)
        original_cwd = Path.cwd()
        monkeypatch.setattr(sys, "path", original_sys_path.copy())
        try:
            runner._inject_sam_body4d_path(repo)
            for sub in (
                str(repo),
                str(repo / "scripts"),
                str(repo / "models" / "sam_3d_body"),
                str(repo / "models" / "diffusion_vas"),
            ):
                assert sub in sys.path
            assert Path.cwd().resolve() == repo.resolve()
        finally:
            os.chdir(original_cwd)

    def test_idempotent_does_not_duplicate_entries(self, tmp_path: Path, monkeypatch):
        repo = tmp_path / "sam-body4d"
        (repo / "scripts").mkdir(parents=True)
        (repo / "models" / "sam_3d_body").mkdir(parents=True)
        (repo / "models" / "diffusion_vas").mkdir(parents=True)
        original_sys_path = list(sys.path)
        original_cwd = Path.cwd()
        monkeypatch.setattr(sys, "path", original_sys_path.copy())
        try:
            runner._inject_sam_body4d_path(repo)
            runner._inject_sam_body4d_path(repo)
            assert sys.path.count(str(repo)) == 1
            assert sys.path.count(str(repo / "scripts")) == 1
        finally:
            os.chdir(original_cwd)

    def test_missing_repo_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="SAM-Body4D repo"):
            runner._inject_sam_body4d_path(tmp_path / "does_not_exist")


class TestPatchConfig:
    def _write_base_config(self, p: Path) -> Path:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "paths:\n  ckpt_root: /unused\n"
            "completion:\n  enable: true\n"
            "sam_3d_body:\n  batch_size: 64\n  ckpt_path: /a\n  mhr_path: /b\n  fov_path: /c\n  detector_path: \"\"\n  segmentor_path: \"\"\n"
            "runtime:\n  output_dir: ./outputs\n"
        )
        return p

    def test_disable_completion_and_batch_override(self, tmp_path: Path):
        omegaconf = pytest.importorskip("omegaconf")
        base = self._write_base_config(tmp_path / "base.yaml")
        out = tmp_path / "wd"
        patched_path = runner._patch_config(
            base_config_path=base,
            out_dir=out,
            disable_completion=True,
            batch_size=16,
        )
        assert patched_path == out / "_runtime_config.yaml"
        cfg = omegaconf.OmegaConf.load(str(patched_path))
        assert cfg.completion.enable is False
        assert cfg.sam_3d_body.batch_size == 16
        assert cfg.runtime.output_dir == str(out)

    def test_keeps_completion_when_not_disabled(self, tmp_path: Path):
        omegaconf = pytest.importorskip("omegaconf")
        base = self._write_base_config(tmp_path / "base.yaml")
        out = tmp_path / "wd"
        patched_path = runner._patch_config(base, out, disable_completion=False, batch_size=None)
        cfg = omegaconf.OmegaConf.load(str(patched_path))
        assert cfg.completion.enable is True
        assert cfg.sam_3d_body.batch_size == 64

    def test_creates_out_dir_if_missing(self, tmp_path: Path):
        pytest.importorskip("omegaconf")
        base = self._write_base_config(tmp_path / "base.yaml")
        out = tmp_path / "deeply" / "nested" / "wd"
        runner._patch_config(base, out, disable_completion=False, batch_size=None)
        assert out.is_dir()
