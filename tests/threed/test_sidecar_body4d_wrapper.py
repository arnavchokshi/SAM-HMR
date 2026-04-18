"""Unit tests for :mod:`threed.sidecar_body4d.wrapper` (plan Task 9)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from threed.sidecar_body4d.wrapper import (
    consolidate_joints_npy,
    intermediates_layout_ok,
    iter_palette_obj_ids,
    link_artifacts_into_workdir,
    monkeypatch_sam3,
    monkeypatch_save_mesh_results,
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


class TestMonkeypatchSaveMeshResults:
    def _build_fake_module(self) -> types.ModuleType:
        m = types.ModuleType("fake_offline_app")
        m._calls = []
        def real_save(outputs, faces, save_dir, focal_dir, image_path, id_current):
            m._calls.append({
                "outputs": outputs, "faces": faces,
                "save_dir": save_dir, "focal_dir": focal_dir,
                "image_path": image_path, "id_current": id_current,
            })
        m.save_mesh_results = real_save
        return m

    def test_calls_original_first(self, tmp_path: Path):
        m = self._build_fake_module()
        joints_dir = tmp_path / "joints"
        monkeypatch_save_mesh_results(m, joints_dir)
        outputs = [
            {"pred_keypoints_3d": np.arange(70 * 3).reshape(70, 3).astype(np.float32)}
        ]
        m.save_mesh_results(
            outputs, faces="FACES", save_dir="/tmp/save", focal_dir="/tmp/focal",
            image_path="/tmp/00000007.jpg", id_current=[2],
        )
        assert len(m._calls) == 1, "original save_mesh_results MUST still be called"
        assert m._calls[0]["image_path"] == "/tmp/00000007.jpg"

    def test_writes_one_npy_per_person(self, tmp_path: Path):
        m = self._build_fake_module()
        joints_dir = tmp_path / "joints"
        monkeypatch_save_mesh_results(m, joints_dir)
        outputs = [
            {"pred_keypoints_3d": np.full((70, 3), 1.0, dtype=np.float32)},
            {"pred_keypoints_3d": np.full((70, 3), 2.0, dtype=np.float32)},
        ]
        m.save_mesh_results(
            outputs, faces=None, save_dir="/tmp/s", focal_dir="/tmp/f",
            image_path="/tmp/clip/00000042.jpg", id_current=[0, 4],
        )
        f1 = joints_dir / "1" / "00000042.npy"
        f5 = joints_dir / "5" / "00000042.npy"
        assert f1.is_file() and f5.is_file()
        np.testing.assert_array_equal(np.load(f1), np.full((70, 3), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(np.load(f5), np.full((70, 3), 2.0, dtype=np.float32))

    def test_idempotent(self, tmp_path: Path):
        m = self._build_fake_module()
        monkeypatch_save_mesh_results(m, tmp_path / "joints")
        first = m.save_mesh_results
        monkeypatch_save_mesh_results(m, tmp_path / "joints")
        assert m.save_mesh_results is first, (
            "second monkeypatch must reuse the existing wrapper"
        )

    def test_empty_outputs_noop(self, tmp_path: Path):
        m = self._build_fake_module()
        joints_dir = tmp_path / "joints"
        monkeypatch_save_mesh_results(m, joints_dir)
        m.save_mesh_results([], None, "/s", "/f", "/00000000.jpg", [])
        assert not joints_dir.exists() or not list(joints_dir.iterdir())

    def test_skips_when_pred_keypoints_3d_missing(self, tmp_path: Path, capsys):
        """If pred_keypoints_3d isn't present, log a warning and skip without crashing.

        Defensive: if upstream's save_mesh_results signature ever changes
        and stops passing keypoints, we shouldn't break the run.
        """
        m = self._build_fake_module()
        joints_dir = tmp_path / "joints"
        monkeypatch_save_mesh_results(m, joints_dir)
        m.save_mesh_results(
            [{"pred_vertices": np.zeros((100, 3), dtype=np.float32)}],
            None, "/s", "/f", "/00000000.jpg", [0],
        )
        captured = capsys.readouterr()
        assert "pred_keypoints_3d" in (captured.out + captured.err)


class TestConsolidateJointsNpy:
    def test_packs_per_track_per_frame(self, tmp_path: Path):
        joints_dir = tmp_path / "joints"
        for tid in (1, 3):
            (joints_dir / str(tid)).mkdir(parents=True)
        for f in (0, 1, 2):
            np.save(joints_dir / "1" / f"{f:08d}.npy",
                    np.full((70, 3), float(f), dtype=np.float32))
            np.save(joints_dir / "3" / f"{f:08d}.npy",
                    np.full((70, 3), float(f) + 100, dtype=np.float32))
        out = consolidate_joints_npy(joints_dir, tids=[1, 3], n_frames=3, n_joints=70)
        assert out.shape == (3, 2, 70, 3)
        np.testing.assert_array_equal(out[2, 0, 0], np.full(3, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(out[2, 1, 0], np.full(3, 102.0, dtype=np.float32))

    def test_missing_frames_become_nan(self, tmp_path: Path):
        joints_dir = tmp_path / "joints"
        (joints_dir / "1").mkdir(parents=True)
        np.save(joints_dir / "1" / "00000000.npy",
                np.zeros((70, 3), dtype=np.float32))
        out = consolidate_joints_npy(joints_dir, tids=[1], n_frames=3, n_joints=70)
        assert out.shape == (3, 1, 70, 3)
        assert not np.isnan(out[0]).any()
        assert np.isnan(out[1]).all()
        assert np.isnan(out[2]).all()

    def test_missing_track_dir_all_nan(self, tmp_path: Path):
        joints_dir = tmp_path / "joints"
        joints_dir.mkdir()
        out = consolidate_joints_npy(joints_dir, tids=[1, 2], n_frames=2, n_joints=70)
        assert out.shape == (2, 2, 70, 3)
        assert np.isnan(out).all()


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
