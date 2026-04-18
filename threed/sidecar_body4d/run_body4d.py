"""SAM-Body4D sidecar runner — Stage C2 of the dual 3D pipeline.

Consumes the artifacts produced by Tasks 1-6 (frames_full + tracks +
SAM-2 palette masks) and delegates the actual mesh recovery to
SAM-Body4D's :class:`OfflineApp`. Per plan §3.4 we **bypass SAM-3
entirely** — our DeepOcSort + SAM-2 masks are already per-frame palette
PNGs (pixel == tid), which is exactly what SAM-Body4D's
``process_image_with_mask`` consumes.

Reads from ``--intermediates-dir``::

    frames_full/                    # original-resolution JPGs
    masks_palette/<frame:08d>.png   # palette PNG, pixel == tid
    tracks.pkl                      # for the tid list

Writes under ``--output-dir`` (matching plan §4.1's ``sam_body4d/`` contract)::

    images/                         # symlinks back to frames_full (workdir)
    masks/                          # symlinks back to masks_palette (workdir)
    mesh_4d_individual/<tid>/<frame:08d>.ply
    focal_4d_individual/<tid>/<frame:08d>.json
    rendered_frames/<frame:08d>.jpg
    rendered_frames_individual/<tid>/<frame:08d>.jpg
    4d_<gen_id>.mp4
    run_summary.json                # written by this runner: timings, VRAM, paths

Joint extraction (per-frame COCO-17 from each PLY + the MHR regressor)
is deferred to Stage D so the same logic runs on SMPL-X too, keeping
the comparison code authoritative for joint conventions.

The wrapper module (``threed.sidecar_body4d.wrapper``) carries the
GPU-free helpers and is fully unit-tested. This runner is the GPU-side
glue and is validated by the box-side smoke test
(``~/work/run_task10_smoke.sh``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from threed.io import load_tracks
from threed.sidecar_body4d.wrapper import (
    intermediates_layout_ok,
    iter_palette_obj_ids,
    link_artifacts_into_workdir,
    monkeypatch_sam3,
    sorted_tid_list,
    workdir_layout_ok,
)


def _inject_sam_body4d_path(sam_body4d_path: Path) -> None:
    """Add SAM-Body4D's import roots to ``sys.path`` and chdir into it.

    SAM-Body4D's ``scripts/offline_app.py`` does ``from utils import ...``
    relative to the repo root, and ``models/sam_3d_body`` + 
    ``models/diffusion_vas`` both have their own top-level packages.
    The order matters: the repo root must come BEFORE the per-model
    subdirs so that ``from utils import ...`` resolves to
    ``<repo>/utils.py`` and not ``<repo>/models/diffusion_vas/utils.py``
    (which has a different signature). We also ``chdir`` so any relative
    paths inside config files resolve against the repo root.
    """
    sam_body4d_path = Path(sam_body4d_path).expanduser().resolve()
    if not sam_body4d_path.is_dir():
        raise FileNotFoundError(f"SAM-Body4D repo not found at {sam_body4d_path}")
    os.chdir(sam_body4d_path)
    for sub in (
        sam_body4d_path,
        sam_body4d_path / "scripts",
        sam_body4d_path / "models" / "sam_3d_body",
        sam_body4d_path / "models" / "diffusion_vas",
    ):
        s = str(sub)
        if s not in sys.path:
            sys.path.append(s)


def _patch_config(
    base_config_path: Path,
    out_dir: Path,
    disable_completion: bool,
    batch_size: Optional[int],
) -> Path:
    """Write a clip-local copy of body4d.yaml with our overrides.

    Two production knobs (per plan §3.5):

    - ``disable_completion``: drop the Diffusion-VAS branch (~9× speedup,
      ~halves VRAM). Used for fast iteration / ``adiTest`` smokes; we
      keep it ON for the final ``loveTest`` run where occlusion handling
      matters.
    - ``batch_size``: defaults to 64 (max throughput). Drop to 32 or 16
      to fit dense scenes on a 40 GB A100; resources.md table:
      5 dancers + completion + batch=32 = 35 GB peak; batch=64 OOMs.

    The patched YAML is written to ``out_dir/_runtime_config.yaml`` so
    each run has its own pinned config (helps when re-running with
    different overrides without disturbing the box-side ``configs/``).
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(base_config_path))
    if disable_completion:
        cfg.completion.enable = False
    if batch_size is not None:
        cfg.sam_3d_body.batch_size = int(batch_size)
    cfg.runtime.output_dir = str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    patched = out_dir / "_runtime_config.yaml"
    OmegaConf.save(cfg, str(patched))
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediates-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sam-body4d-path",
        type=Path,
        default=Path("~/code/sam-body4d").expanduser(),
        help="Clone of github.com/gaomingqi/sam-body4d (default: ~/code/sam-body4d)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to body4d.yaml (default: <sam-body4d-path>/configs/body4d.yaml)",
    )
    parser.add_argument(
        "--disable-completion",
        action="store_true",
        help="Skip the Diffusion-VAS occlusion branch (faster, lower VRAM).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override sam_3d_body.batch_size (default: from config, usually 64)",
    )
    args = parser.parse_args()

    interm: Path = args.intermediates_dir.expanduser().resolve()
    out_dir: Path = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, errs = intermediates_layout_ok(interm)
    if not ok:
        for e in errs:
            print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    tracks = load_tracks(interm / "tracks.pkl")
    tids = sorted_tid_list(tracks)
    iter_palette_obj_ids(tids)
    print(f"[ok] {len(tids)} tracks: {tids}")

    sam_body4d_path = args.sam_body4d_path.expanduser().resolve()
    _inject_sam_body4d_path(sam_body4d_path)

    config_path = (
        args.config_path.expanduser().resolve()
        if args.config_path is not None
        else sam_body4d_path / "configs" / "body4d.yaml"
    )
    patched_cfg = _patch_config(
        base_config_path=config_path,
        out_dir=out_dir,
        disable_completion=args.disable_completion,
        batch_size=args.batch_size,
    )
    print(f"[ok] runtime config: {patched_cfg}")

    n_frames, n_masks = link_artifacts_into_workdir(
        out_dir,
        interm / "frames_full",
        interm / "masks_palette",
    )
    print(f"[ok] linked {n_frames} frames + {n_masks} palette masks into {out_dir}")
    ok, errs = workdir_layout_ok(out_dir)
    if not ok:
        for e in errs:
            print(f"[ERROR] {e}", file=sys.stderr)
        return 3

    import scripts.offline_app as oa  # noqa: E402 — needs sys.path injection first
    import torch  # noqa: E402

    monkeypatch_sam3(oa)
    print("[ok] SAM-3 builder monkey-patched (we use SAM-2 masks instead)")

    torch.cuda.reset_peak_memory_stats()
    t_init = time.time()
    app = oa.OfflineApp(config_path=str(patched_cfg))
    app.OUTPUT_DIR = str(out_dir)
    app.RUNTIME["out_obj_ids"] = tids
    init_seconds = time.time() - t_init
    init_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[ok] OfflineApp init in {init_seconds:.1f} s, "
        f"VRAM after init={init_vram_gb:.2f} GB"
    )

    torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    with torch.autocast("cuda", enabled=False):
        app.on_4d_generation()
    run_seconds = time.time() - t_run
    run_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[ok] on_4d_generation in {run_seconds:.1f} s, "
        f"peak VRAM during run={run_vram_gb:.2f} GB"
    )

    n_plys = sum(1 for _ in (out_dir / "mesh_4d_individual").glob("*/*.ply"))
    n_focals = sum(1 for _ in (out_dir / "focal_4d_individual").glob("*/*.json"))
    n_rendered = sum(1 for _ in (out_dir / "rendered_frames").glob("*.jpg"))
    n_mp4 = sum(1 for _ in out_dir.glob("4d_*.mp4"))
    print(
        f"[done] outputs: {n_plys} PLYs, {n_focals} focal JSONs, "
        f"{n_rendered} rendered frames, {n_mp4} 4D MP4s"
    )

    summary = {
        "tids": tids,
        "n_frames": n_frames,
        "init_seconds": round(init_seconds, 2),
        "init_vram_gb": round(init_vram_gb, 2),
        "run_seconds": round(run_seconds, 2),
        "run_vram_gb": round(run_vram_gb, 2),
        "n_ply_meshes": n_plys,
        "n_focal_jsons": n_focals,
        "n_rendered_frames": n_rendered,
        "n_4d_mp4": n_mp4,
        "config": {
            "completion_enabled": not args.disable_completion,
            "batch_size_override": args.batch_size,
            "patched_config_path": str(patched_cfg),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[wrote] {out_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
