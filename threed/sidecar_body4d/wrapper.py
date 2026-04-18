"""GPU-free helpers for the SAM-Body4D sidecar (plan Task 9).

This module is the integration glue between our shared intermediates
(frames + tracks + per-track + palette masks produced by Tasks 1-6) and
SAM-Body4D's :class:`OfflineApp`. Two responsibilities:

1. **Monkey-patch SAM-3 out** — :func:`monkeypatch_sam3` replaces
   ``scripts.offline_app.build_sam3_from_config`` with a no-op so we
   never load ``sam3.pt`` (we already have masks from upstream
   DeepOcSort + SAM-2; loading SAM-3 would just waste 8 GB of VRAM and
   ~70 s of boot time, and it requires a HF gate that may be PENDING).
   This is the production design described in plan §3.4.

2. **Materialise palette masks + frames into OfflineApp's working dir**
   — :func:`link_artifacts_into_workdir` symlinks frames_full and
   masks_palette into ``OUTPUT_DIR/{images,masks}/``. SAM-Body4D's
   ``on_4d_generation`` re-reads them via ``glob.glob``; symlinks are
   followed transparently and avoid duplicating ~hundreds of MB of
   JPGs/PNGs per clip.

The runner itself (Stage C2) lives in ``threed.sidecar_body4d.run_body4d``
(plan Task 10). All helpers here are import-safe in any conda env (no
torch, no cv2, no SAM-Body4D imports) so they can be unit-tested from
the host repo without the body4d env present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


_SAM3_PATCH_SENTINEL = "_sam3_patched_by_threed"
_JOINTS_PATCH_SENTINEL = "_save_mesh_joints_patched_by_threed"


def monkeypatch_sam3(offline_app_module: Any) -> None:
    """Replace ``offline_app_module.build_sam3_from_config`` with a no-op.

    The replacement returns ``(None, None)`` so that
    ``OfflineApp.__init__`` happily assigns
    ``self.sam3_model = None; self.predictor = None``. Neither attribute
    is accessed by ``on_4d_generation``, which is the only entry point
    we use.

    The function is **idempotent** — re-applying it on an already-patched
    module is a no-op, so it is safe to call from multiple sidecars or
    from a script that is rerun. We mark the patched module via
    ``getattr(module, _SAM3_PATCH_SENTINEL, False)`` rather than
    inspecting the bytecode of the replacement (which would be brittle
    against pickling/decorators).
    """
    if getattr(offline_app_module, _SAM3_PATCH_SENTINEL, False):
        return

    def _no_sam3(_cfg):  # noqa: ANN001 — config object, never inspected
        return None, None

    offline_app_module.build_sam3_from_config = _no_sam3
    setattr(offline_app_module, _SAM3_PATCH_SENTINEL, True)


def intermediates_layout_ok(interm: Path) -> Tuple[bool, List[str]]:
    """Check the artifact layout produced by Tasks 1-6 for SAM-Body4D.

    The body4d input set differs from PromptHMR's:

    - SAM-Body4D wants ``frames_full/`` (original resolution; SAM-3D-Body
      handles arbitrary sizes per plan §3.3) — NOT the 896-cap ``frames/``
      that PromptHMR uses.
    - SAM-Body4D wants ``masks_palette/`` (palette PNGs, pixel == tid) —
      this is what ``process_image_with_mask`` consumes directly.
    - ``tracks.pkl`` is needed only for the tid list (we read it lazily;
      ``masks`` and ``detected`` slots are not consumed by SAM-Body4D).

    Returns ``(ok, errors)`` so callers can show every missing artifact
    in one shot, mirroring the contract used by the PromptHMR sidecar.
    The frame count and palette-mask count MUST match (they are written
    in lock-step by Task 6); a mismatch is a hard error because
    SAM-Body4D's ``on_4d_generation`` zips ``images_list`` with
    ``masks_list`` and would silently process a truncated stream.
    """
    errs: List[str] = []
    interm = Path(interm)

    frames_dir = interm / "frames_full"
    if not frames_dir.is_dir():
        errs.append(f"frames_full/ directory missing under {interm}")
        n_frames = 0
    else:
        frames = sorted(frames_dir.glob("*.jpg"))
        n_frames = len(frames)
        if n_frames == 0:
            errs.append(f"frames_full/ has no JPG files under {interm}")

    masks_dir = interm / "masks_palette"
    if not masks_dir.is_dir():
        errs.append(f"masks_palette/ directory missing under {interm}")
        n_masks = 0
    else:
        masks = sorted(masks_dir.glob("*.png"))
        n_masks = len(masks)
        if n_masks == 0:
            errs.append(f"masks_palette/ has no PNG files under {interm}")

    if not (interm / "tracks.pkl").is_file():
        errs.append(f"tracks.pkl missing under {interm}")

    if n_frames > 0 and n_masks > 0 and n_frames != n_masks:
        errs.append(
            f"frame/mask count mismatch under {interm}: "
            f"frames_full={n_frames}, masks_palette={n_masks}"
        )

    return len(errs) == 0, errs


def sorted_tid_list(tracks: Mapping) -> List[int]:
    """Stable, python-int sort of track IDs.

    joblib emits ``np.int64`` keys on round-trip, which break naive
    ``sorted(tracks.keys())`` chained into ``json.dumps`` downstream
    (numpy ints are not JSON-serialisable). We coerce to python int
    explicitly so callers can pass the result to OmegaConf and JSON
    without further casts. Mirrors the helper of the same name in
    ``threed.sidecar_promthmr.run_promthmr_vid`` — duplicated rather
    than shared because the test boundary is per-sidecar.
    """
    return sorted(int(k) for k in tracks.keys())


def link_artifacts_into_workdir(
    out_dir: Path,
    frames_full_dir: Path,
    masks_palette_dir: Path,
) -> Tuple[int, int]:
    """Symlink ``frames_full/*.jpg`` and ``masks_palette/*.png`` into ``out_dir``.

    SAM-Body4D's :meth:`OfflineApp.on_4d_generation` reads ``self.OUTPUT_DIR/images``
    and ``self.OUTPUT_DIR/masks`` via ``glob.glob`` (see
    ``scripts/offline_app.py:223-247``). Symlinks are followed
    transparently by both ``glob`` and ``cv2.imread``, so we avoid
    duplicating hundreds of MB of JPGs/PNGs per clip.

    Filenames are preserved (Task 6 already names them ``{frame:08d}.jpg``
    / ``{frame:08d}.png``, which is what ``glob`` then sorts
    correctly). Existing entries inside ``out_dir/images`` or
    ``out_dir/masks`` are removed first so this function is safe to
    re-run on a dirty workdir.

    Raises ``ValueError`` if the source frame count does not equal the
    source mask count — see :func:`intermediates_layout_ok` for the
    motivation.
    """
    out_dir = Path(out_dir)
    frames_full_dir = Path(frames_full_dir)
    masks_palette_dir = Path(masks_palette_dir)

    images_out = out_dir / "images"
    masks_out = out_dir / "masks"

    for d in (images_out, masks_out):
        if d.exists():
            for child in d.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
        d.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_full_dir.glob("*.jpg"))
    masks = sorted(masks_palette_dir.glob("*.png"))
    if len(frames) != len(masks):
        raise ValueError(
            f"frame/mask count mismatch: {len(frames)} frames vs {len(masks)} masks"
        )

    for src in frames:
        (images_out / src.name).symlink_to(src.resolve())
    for src in masks:
        (masks_out / src.name).symlink_to(src.resolve())

    return len(frames), len(masks)


def workdir_layout_ok(out_dir: Path) -> Tuple[bool, List[str]]:
    """Post-link sanity check that ``OfflineApp.on_4d_generation`` won't choke.

    Mirrors the upstream glob in ``offline_app.py:223-247`` and confirms
    we have a non-empty ``images/`` AND a non-empty ``masks/`` AND that
    they are 1:1 by basename (frame N has both
    ``images/00000N.jpg`` and ``masks/00000N.png``). Useful as a
    pre-flight inside the runner so an early failure surfaces before
    the (slow) ``OfflineApp.__init__`` GPU load.
    """
    errs: List[str] = []
    out_dir = Path(out_dir)
    images = sorted((out_dir / "images").glob("*.jpg"))
    masks = sorted((out_dir / "masks").glob("*.png"))
    if not images:
        errs.append(f"{out_dir}/images/ has no JPG files")
    if not masks:
        errs.append(f"{out_dir}/masks/ has no PNG files")
    img_stems = {p.stem for p in images}
    msk_stems = {p.stem for p in masks}
    if img_stems and msk_stems and img_stems != msk_stems:
        only_img = sorted(img_stems - msk_stems)[:3]
        only_msk = sorted(msk_stems - img_stems)[:3]
        errs.append(
            f"basename mismatch under {out_dir}: "
            f"only-in-images={only_img}, only-in-masks={only_msk}"
        )
    return len(errs) == 0, errs


def monkeypatch_save_mesh_results(
    offline_app_module: Any,
    joints_dir: Path,
) -> None:
    """Wrap ``offline_app_module.save_mesh_results`` to also dump joints.

    The upstream ``save_mesh_results`` in
    ``models/sam_3d_body/notebook/utils.py`` (imported into
    ``scripts.offline_app``'s namespace) is responsible for writing PLY
    meshes and focal JSONs per (tid, frame). It receives
    ``person_output["pred_keypoints_3d"]`` (the MHR70 3D keypoints —
    shape ``(70, 3)``) but never persists them.

    Our wrapper:

    1. Calls the original first (PLY + focal JSON outputs unchanged).
    2. For each ``person_output``, writes
       ``joints_dir/<tid>/<frame:08d>.npy`` with
       ``pred_keypoints_3d`` as ``np.float32``.

    Track ID mirrors the upstream PLY layout: ``tid = id_current[pid] + 1``
    so the joint files line up with ``mesh_4d_individual/<tid>/`` for
    later consolidation by :func:`consolidate_joints_npy`.

    Idempotent (sentinel-marked, same pattern as :func:`monkeypatch_sam3`).
    Defensive — if a future upstream change drops ``pred_keypoints_3d``
    from ``outputs`` we log a warning per-frame and continue (PLYs are
    still saved by the original).
    """
    if getattr(offline_app_module, _JOINTS_PATCH_SENTINEL, False):
        return

    original = offline_app_module.save_mesh_results
    joints_dir = Path(joints_dir)

    def _wrapped(outputs, faces, save_dir, focal_dir, image_path, id_current):
        original(outputs, faces, save_dir, focal_dir, image_path, id_current)
        if not outputs:
            return
        import os
        base = os.path.basename(image_path)[:-4]
        for pid, person_output in enumerate(outputs):
            kp = person_output.get("pred_keypoints_3d")
            if kp is None:
                print(
                    f"[wrapper] WARN: pred_keypoints_3d missing for pid={pid} "
                    f"frame={base}; skipping joint dump"
                )
                continue
            tid = int(id_current[pid]) + 1
            out_dir = joints_dir / str(tid)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{base}.npy", np.asarray(kp, dtype=np.float32))

    offline_app_module.save_mesh_results = _wrapped
    setattr(offline_app_module, _JOINTS_PATCH_SENTINEL, True)


def consolidate_joints_npy(
    joints_dir: Path,
    tids: Sequence[int],
    n_frames: int,
    n_joints: int = 70,
) -> np.ndarray:
    """Pack per-(tid, frame) joint dumps into a single ``(T, N, J, 3)`` array.

    Reads ``joints_dir/<tid>/<frame:08d>.npy`` for every (frame in
    ``range(n_frames)``, tid in ``tids``) and stacks them into a single
    NaN-padded array. Missing files become NaN — matches the convention
    used by ``threed.sidecar_promthmr.run_promthmr_vid.joints_world_padded``
    so Stage D's ``np.nanmean`` semantics work uniformly across pipelines.

    The dancer axis (``N``) is ordered by ``tids`` (use the canonical
    sorted list). Joint count is fixed at MHR70 = 70 by default; pass
    ``n_joints=17`` if upstream pre-reduces to COCO-17 (we don't).
    """
    joints_dir = Path(joints_dir)
    out = np.full((n_frames, len(tids), n_joints, 3), np.nan, dtype=np.float32)
    for di, tid in enumerate(tids):
        sub = joints_dir / str(int(tid))
        if not sub.is_dir():
            continue
        for f in range(n_frames):
            p = sub / f"{f:08d}.npy"
            if p.is_file():
                out[f, di] = np.load(p)
    return out


def iter_palette_obj_ids(track_ids: Iterable[int]) -> List[int]:
    """Validate that all track IDs fit in a palette PNG (1..255).

    Palette PNGs only encode 256 levels; index 0 is reserved for
    background, so we have 255 distinct foreground ids. Our DeepOcSort
    emits stable integer ids from 1; in practice clips top out at ~15
    dancers (loveTest), so this is a defensive guard rather than a
    hot constraint. Returns the de-duped, python-int sorted list and
    raises ``ValueError`` if any id is out of range or non-positive.
    """
    out = sorted({int(t) for t in track_ids})
    for tid in out:
        if tid < 1 or tid > 255:
            raise ValueError(
                f"track id {tid} out of palette PNG range [1..255]"
            )
    return out
