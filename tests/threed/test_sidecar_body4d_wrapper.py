"""Unit tests for :mod:`threed.sidecar_body4d.wrapper` (plan Task 9)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from threed.sidecar_body4d.wrapper import (
    intermediates_layout_ok,
    iter_palette_obj_ids,
    link_artifacts_into_workdir,
    monkeypatch_sam3,
    sorted_tid_list,
    workdir_layout_ok,
)


@pytest.fixture
def fake_intermediates(tmp_path: Path) -> Path:
    """Build a minimal-but-valid intermediates layout (2 frames, 2 tids)."""
    interm = tmp_path / "interm"
    (interm / "frames_full").mkdir(parents=True)
    (interm / "masks_palette").mkdir(parents=True)
    (interm / "frames_full" / "00000000.jpg").write_bytes(b"jpg-stub-0")
    (interm / "frames_full" / "00000001.jpg").write_bytes(b"jpg-stub-1")
    (interm / "masks_palette" / "00000000.png").write_bytes(b"png-stub-0")
    (interm / "masks_palette" / "00000001.png").write_bytes(b"png-stub-1")
    (interm / "tracks.pkl").write_bytes(b"joblib-stub")
    return interm


class TestMonkeypatchSam3:
    def _build_fake_module(self) -> types.ModuleType:
        m = types.ModuleType("fake_offline_app")
        def real_builder(cfg):  # noqa: ANN001 — stub
            raise RuntimeError("real SAM-3 loader was called")
        m.build_sam3_from_config = real_builder
        return m

    def test_replaces_builder_with_no_op(self):
        m = self._build_fake_module()
        monkeypatch_sam3(m)
        out = m.build_sam3_from_config({"sam3": {"ckpt_path": "/never/loaded"}})
        assert out == (None, None)

    def test_idempotent(self):
        m = self._build_fake_module()
        monkeypatch_sam3(m)
        first_replacement = m.build_sam3_from_config
        monkeypatch_sam3(m)
        assert m.build_sam3_from_config is first_replacement, (
            "second monkeypatch must reuse the existing replacement, not "
            "re-wrap an already-patched module"
        )

    def test_marks_module_with_sentinel(self):
        m = self._build_fake_module()
        assert not getattr(m, "_sam3_patched_by_threed", False)
        monkeypatch_sam3(m)
        assert getattr(m, "_sam3_patched_by_threed", False) is True

    def test_does_not_clobber_other_module_attrs(self):
        m = self._build_fake_module()
        m.OfflineApp = "sentinel"
        m.RUNTIME = {"out_obj_ids": [1, 2, 3]}
        monkeypatch_sam3(m)
        assert m.OfflineApp == "sentinel"
        assert m.RUNTIME == {"out_obj_ids": [1, 2, 3]}


class TestIntermediatesLayoutOk:
    def test_happy_path(self, fake_intermediates: Path):
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert ok is True, errs
        assert errs == []

    def test_missing_frames_full(self, fake_intermediates: Path):
        for f in (fake_intermediates / "frames_full").iterdir():
            f.unlink()
        (fake_intermediates / "frames_full").rmdir()
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert not ok
        assert any("frames_full/" in e for e in errs)

    def test_empty_frames_full(self, fake_intermediates: Path):
        for f in (fake_intermediates / "frames_full").iterdir():
            f.unlink()
        for f in (fake_intermediates / "masks_palette").iterdir():
            f.unlink()
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert not ok
        assert any("no JPG files" in e for e in errs)

    def test_missing_masks_palette(self, fake_intermediates: Path):
        for f in (fake_intermediates / "masks_palette").iterdir():
            f.unlink()
        (fake_intermediates / "masks_palette").rmdir()
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert not ok
        assert any("masks_palette/" in e for e in errs)

    def test_missing_tracks_pkl(self, fake_intermediates: Path):
        (fake_intermediates / "tracks.pkl").unlink()
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert not ok
        assert any("tracks.pkl" in e for e in errs)

    def test_count_mismatch(self, fake_intermediates: Path):
        (fake_intermediates / "masks_palette" / "00000002.png").write_bytes(b"extra")
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert not ok
        assert any("count mismatch" in e and "frames_full=2" in e and "masks_palette=3" in e for e in errs)

    def test_aggregates_all_errors(self, fake_intermediates: Path):
        for f in (fake_intermediates / "frames_full").iterdir():
            f.unlink()
        (fake_intermediates / "tracks.pkl").unlink()
        ok, errs = intermediates_layout_ok(fake_intermediates)
        assert not ok
        assert len(errs) >= 2


class TestSortedTidList:
    def test_basic(self):
        assert sorted_tid_list({3: "x", 1: "y", 2: "z"}) == [1, 2, 3]

    def test_numpy_int_keys(self):
        d = {np.int64(7): "a", np.int64(2): "b", np.int64(11): "c"}
        out = sorted_tid_list(d)
        assert out == [2, 7, 11]
        assert all(type(t) is int for t in out)

    def test_empty(self):
        assert sorted_tid_list({}) == []

    def test_mixed_int_and_numpy(self):
        d = {1: "a", np.int64(5): "b", 3: "c"}
        assert sorted_tid_list(d) == [1, 3, 5]


class TestLinkArtifactsIntoWorkdir:
    def test_creates_symlinks_to_real_files(self, fake_intermediates: Path, tmp_path: Path):
        out_dir = tmp_path / "workdir"
        n_f, n_m = link_artifacts_into_workdir(
            out_dir,
            fake_intermediates / "frames_full",
            fake_intermediates / "masks_palette",
        )
        assert n_f == 2 and n_m == 2
        for name in ("00000000.jpg", "00000001.jpg"):
            link = out_dir / "images" / name
            assert link.is_symlink()
            assert link.resolve() == (fake_intermediates / "frames_full" / name).resolve()
        for name in ("00000000.png", "00000001.png"):
            link = out_dir / "masks" / name
            assert link.is_symlink()
            assert link.resolve() == (fake_intermediates / "masks_palette" / name).resolve()

    def test_count_mismatch_raises(self, fake_intermediates: Path, tmp_path: Path):
        (fake_intermediates / "masks_palette" / "00000002.png").write_bytes(b"extra")
        with pytest.raises(ValueError, match="count mismatch"):
            link_artifacts_into_workdir(
                tmp_path / "wd",
                fake_intermediates / "frames_full",
                fake_intermediates / "masks_palette",
            )

    def test_idempotent_overwrites_existing(self, fake_intermediates: Path, tmp_path: Path):
        out = tmp_path / "wd"
        link_artifacts_into_workdir(out, fake_intermediates / "frames_full", fake_intermediates / "masks_palette")
        link_artifacts_into_workdir(out, fake_intermediates / "frames_full", fake_intermediates / "masks_palette")
        assert sorted(p.name for p in (out / "images").iterdir()) == ["00000000.jpg", "00000001.jpg"]
        assert sorted(p.name for p in (out / "masks").iterdir()) == ["00000000.png", "00000001.png"]

    def test_creates_parent_dirs(self, fake_intermediates: Path, tmp_path: Path):
        out = tmp_path / "deeply" / "nested" / "wd"
        n_f, n_m = link_artifacts_into_workdir(
            out, fake_intermediates / "frames_full", fake_intermediates / "masks_palette"
        )
        assert n_f == 2 and n_m == 2
        assert (out / "images").is_dir()
        assert (out / "masks").is_dir()


class TestWorkdirLayoutOk:
    def test_happy_path(self, fake_intermediates: Path, tmp_path: Path):
        out = tmp_path / "wd"
        link_artifacts_into_workdir(out, fake_intermediates / "frames_full", fake_intermediates / "masks_palette")
        ok, errs = workdir_layout_ok(out)
        assert ok, errs

    def test_missing_images(self, tmp_path: Path):
        out = tmp_path / "wd"
        (out / "images").mkdir(parents=True)
        (out / "masks").mkdir(parents=True)
        ok, errs = workdir_layout_ok(out)
        assert not ok
        assert any("no JPG" in e for e in errs)

    def test_basename_mismatch(self, tmp_path: Path):
        out = tmp_path / "wd"
        (out / "images").mkdir(parents=True)
        (out / "masks").mkdir(parents=True)
        (out / "images" / "00000000.jpg").write_bytes(b"j")
        (out / "masks" / "00000001.png").write_bytes(b"p")
        ok, errs = workdir_layout_ok(out)
        assert not ok
        assert any("basename mismatch" in e for e in errs)


class TestIterPaletteObjIds:
    def test_basic(self):
        assert iter_palette_obj_ids([3, 1, 2]) == [1, 2, 3]

    def test_dedups(self):
        assert iter_palette_obj_ids([1, 1, 5, 5, 3]) == [1, 3, 5]

    def test_numpy_inputs(self):
        out = iter_palette_obj_ids([np.int64(7), np.int64(7), np.int32(2)])
        assert out == [2, 7]
        assert all(type(t) is int for t in out)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match=r"\[1\.\.255\]"):
            iter_palette_obj_ids([0, 1, 2])

    def test_rejects_over_255(self):
        with pytest.raises(ValueError, match=r"\[1\.\.255\]"):
            iter_palette_obj_ids([1, 256])

    def test_accepts_max_palette(self):
        assert iter_palette_obj_ids([1, 255]) == [1, 255]
