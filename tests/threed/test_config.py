import pytest
from pathlib import Path
from threed.config import CompareConfig, default_config


def test_default_config_has_all_paths(tmp_path):
    cfg = default_config(repo_root=tmp_path)
    assert cfg.repo_root == tmp_path
    assert cfg.output_root == tmp_path / "runs" / "3d_compare"
    assert cfg.phmr_repo.is_absolute()
    assert cfg.body4d_repo.is_absolute()


def test_clip_dirs_are_namespaced(tmp_path):
    cfg = default_config(repo_root=tmp_path)
    dirs = cfg.clip_dirs("loveTest")
    assert dirs.intermediates == tmp_path / "runs" / "3d_compare" / "loveTest" / "intermediates"
    assert dirs.prompthmr == tmp_path / "runs" / "3d_compare" / "loveTest" / "prompthmr"
    assert dirs.sam_body4d == tmp_path / "runs" / "3d_compare" / "loveTest" / "sam_body4d"
    assert dirs.comparison == tmp_path / "runs" / "3d_compare" / "loveTest" / "comparison"
