"""End-to-end driver for the dual 3D pipeline (plan Task 12).

Wraps Stages A → B → C1/C2 → D so a developer can run a single clip
with one command. Stage A and Stage D run in the host conda env
(``threed-host`` / tracking env). Stages B + C1 (PromptHMR-Vid) run in
the ``phmr_pt2.4`` env, and Stage C2 (SAM-Body4D) runs in the
``body4d`` env — they cannot share a Python process because their
torch/cuda/transformers pins are incompatible. Inter-env calls are
issued via ``conda run -n <env> --no-capture-output``.

Stages produced and skip-flag mapping::

    stage_a        — local:    extract frames + tracks         (--skip-stage-a)
    phmr_masks     — phmr env: build SAM-2 masks               (--skip-phmr)
    phmr_run       — phmr env: PromptHMR-Vid HPS               (--skip-phmr)
    phmr_project   — local:    SMPL22 -> COCO17 cam            (--skip-phmr)
    phmr_render    — phmr env: per-frame SMPL-X overlay JPG    (--skip-phmr | --skip-phmr-render)
    body4d         — body4d:   SAM-Body4D + joint dump         (--skip-body4d)
    body4d_render  — body4d:   per-frame mesh overlay on input (--skip-body4d | --skip-body4d-render)
    compare        — local:    metrics.json                    (--skip-compare)
    render         — local:    side_by_side.mp4                (--skip-compare)

The ``render`` step automatically picks up the PHMR overlay JPGs if
``<prompthmr_dir>/rendered_frames/`` exists (i.e. ``phmr_render`` ran
or its artefacts are cached); otherwise the left panel is left blank.
For the right panel it prefers ``<sam_body4d_dir>/rendered_frames_overlay/``
(produced by ``body4d_render``) over upstream's clean-background
``rendered_frames/``, so both panels show meshes overlaid on the
real input video when both render stages have run.

The actual subprocess execution is invoked from ``main()``; the
per-stage command builders + ``plan_pipeline`` are pure and
unit-tested by ``tests/threed/test_orchestrator.py`` so we don't need
a real conda env to validate the wiring.

Smoke testing on the box::

    python scripts/run_3d_compare.py --clip adiTest \\
        --output-root ~/work/3d_compare \\
        --skip-stage-a --skip-phmr \\
        --disable-completion --batch-size 32

(The ``--skip-stage-a`` is needed because the box stages already exist
in ``~/work/3d_compare/<clip>/intermediates/`` from earlier tasks. The
``--output-root`` override points the orchestrator at the box's
``~/work/3d_compare/`` rather than the in-repo ``runs/3d_compare/``.)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from threed.config import default_config

log = logging.getLogger("run_3d_compare")


def build_stage_a_cmd(
    *,
    python: Path,
    clip: str,
    video: Path,
    cache_dir: Path,
    max_frames: Optional[int] = None,
    output_root: Optional[Path] = None,
) -> List[str]:
    """Build the Stage A subprocess command (runs in host env).

    ``max_frames`` (default ``None``) caps how many frames Stage A
    decodes/extracts so downstream PHMR + body4d cost stays bounded for
    long clips. Use this to keep new-clip wall in line with adiTest's
    188-frame baseline.

    ``output_root`` (default ``None``) overrides ``cfg.output_root`` so
    the box can write intermediates under ``~/work/3d_compare/<clip>/``
    rather than the in-repo ``runs/3d_compare/<clip>/``. Mirrors the
    orchestrator's own ``--output-root`` flag.
    """
    cmd = [
        str(python), "-m", "threed.stage_a.run_stage_a",
        "--clip", clip,
        "--video", str(video),
        "--cache-dir", str(cache_dir),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if output_root is not None:
        cmd.extend(["--output-root", str(output_root)])
    return cmd


def build_phmr_masks_cmd(
    *,
    intermediates_dir: Path,
    prompthmr_path: Path,
) -> List[str]:
    """Build the PromptHMR mask-builder subprocess command (runs in phmr env)."""
    return [
        "python", "-m", "threed.sidecar_promthmr.build_masks",
        "--intermediates-dir", str(intermediates_dir),
        "--prompthmr-path", str(prompthmr_path),
    ]


def build_phmr_run_cmd(
    *,
    intermediates_dir: Path,
    output_dir: Path,
    prompthmr_path: Path,
    static_camera: bool,
) -> List[str]:
    """Build the PromptHMR-Vid runner subprocess command (runs in phmr env)."""
    cmd = [
        "python", "-m", "threed.sidecar_promthmr.run_promthmr_vid",
        "--intermediates-dir", str(intermediates_dir),
        "--output-dir", str(output_dir),
        "--prompthmr-path", str(prompthmr_path),
    ]
    if static_camera:
        cmd.append("--static-camera")
    return cmd


def build_phmr_project_joints_cmd(
    *,
    prompthmr_dir: Path,
) -> List[str]:
    """Build the PromptHMR joint projector subprocess command (runs in host env).

    Produces ``<prompthmr_dir>/joints_coco17_cam.npy`` for the
    comparison driver.
    """
    return [
        "python", "-m", "threed.sidecar_promthmr.project_joints",
        "--prompthmr-dir", str(prompthmr_dir),
        "--output", str(prompthmr_dir / "joints_coco17_cam.npy"),
    ]


def build_phmr_render_overlay_cmd(
    *,
    prompthmr_dir: Path,
    frames_dir: Path,
    smplx_path: Path,
) -> List[str]:
    """Build the PromptHMR mesh-overlay renderer command (runs in phmr env).

    Produces ``<prompthmr_dir>/rendered_frames/<frame:08d>.jpg``, which
    the side-by-side renderer auto-detects via ``--prompthmr-frames-dir``
    so the left panel shows real meshes instead of black.
    """
    return [
        "python", "-m", "threed.sidecar_promthmr.render_overlay",
        "--prompthmr-dir", str(prompthmr_dir),
        "--frames-dir", str(frames_dir),
        "--smplx-path", str(smplx_path),
    ]


def build_body4d_cmd(
    *,
    intermediates_dir: Path,
    output_dir: Path,
    sam_body4d_path: Path,
    disable_completion: bool,
    batch_size: Optional[int],
) -> List[str]:
    """Build the SAM-Body4D runner subprocess command (runs in body4d env)."""
    cmd = [
        "python", "-m", "threed.sidecar_body4d.run_body4d",
        "--intermediates-dir", str(intermediates_dir),
        "--output-dir", str(output_dir),
        "--sam-body4d-path", str(sam_body4d_path),
    ]
    if disable_completion:
        cmd.append("--disable-completion")
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    return cmd


def build_body4d_render_overlay_cmd(
    *,
    body4d_dir: Path,
    frames_dir: Path,
) -> List[str]:
    """Build the SAM-Body4D mesh-overlay renderer command (runs in body4d env).

    Produces ``<body4d_dir>/rendered_frames_overlay/<frame:08d>.jpg``,
    which the side-by-side renderer prefers over upstream's
    clean-background ``rendered_frames/`` so the right panel shows real
    meshes overlaid on the dance footage.
    """
    return [
        "python", "-m", "threed.sidecar_body4d.render_overlay",
        "--body4d-dir", str(body4d_dir),
        "--frames-dir", str(frames_dir),
    ]


def build_compare_cmd(
    *,
    prompthmr_joints: Path,
    body4d_joints: Path,
    output: Path,
) -> List[str]:
    """Build the Stage D comparison subprocess command (runs in host env)."""
    return [
        "python", "-m", "threed.compare.run_compare",
        "--prompthmr-joints", str(prompthmr_joints),
        "--body4d-joints", str(body4d_joints),
        "--output", str(output),
    ]


def build_render_cmd(
    *,
    body4d_frames_dir: Path,
    prompthmr_frames_dir: Optional[Path],
    output: Path,
    fps: int,
) -> List[str]:
    """Build the side-by-side renderer subprocess command (runs in host env)."""
    cmd = [
        "python", "-m", "threed.compare.render",
        "--body4d-frames-dir", str(body4d_frames_dir),
        "--output", str(output),
        "--fps", str(fps),
    ]
    if prompthmr_frames_dir is not None:
        cmd += ["--prompthmr-frames-dir", str(prompthmr_frames_dir)]
    return cmd


def plan_pipeline(
    *,
    skip_stage_a: bool,
    skip_phmr: bool,
    skip_body4d: bool,
    skip_compare: bool,
    skip_phmr_render: bool = False,
    skip_body4d_render: bool = False,
) -> List[str]:
    """Return the ordered list of stages that will run, given the skip flags.

    Used by the test suite to verify --skip semantics without invoking
    any subprocess. The returned tags map 1:1 with subprocess
    invocations in :func:`main`.

    ``skip_phmr_render=True`` only suppresses the PHMR mesh-overlay
    render (preserves masks/run/project). ``skip_phmr=True`` is
    dominant — it drops every PHMR stage, including the render.

    ``skip_body4d_render=True`` similarly only suppresses the body4d
    overlay render; ``skip_body4d=True`` drops both body4d stages so
    we don't try to render meshes that were never produced.
    """
    out: List[str] = []
    if not skip_stage_a:
        out.append("stage_a")
    if not skip_phmr:
        out.extend(["phmr_masks", "phmr_run", "phmr_project"])
        if not skip_phmr_render:
            out.append("phmr_render")
    if not skip_body4d:
        out.append("body4d")
        if not skip_body4d_render:
            out.append("body4d_render")
    if not skip_compare:
        out.extend(["compare", "render"])
    return out


def _conda_run(env: str, cmd: Sequence[str], cwd: Path) -> int:
    full = ["conda", "run", "-n", env, "--no-capture-output", *cmd]
    log.info("[%s @ %s] %s", env, cwd, " ".join(full))
    return subprocess.run(full, cwd=str(cwd)).returncode


def _local_run(cmd: Sequence[str], cwd: Path) -> int:
    log.info("[local @ %s] %s", cwd, " ".join(cmd))
    return subprocess.run(list(cmd), cwd=str(cwd)).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--video", type=Path, default=None,
                   help="Source MP4/MOV. Required unless --skip-stage-a.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Stage A cache dir. Required unless --skip-stage-a.")
    p.add_argument("--static-camera", action="store_true",
                   help="Bypass DROID-SLAM in PromptHMR (for tripod clips). "
                        "Only set when SLAM truly fails — see plan §13.")
    p.add_argument("--disable-completion", action="store_true",
                   help="Skip Diffusion-VAS occlusion completion in SAM-Body4D "
                        "(~9× faster, lower VRAM, worse on crowded scenes).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override sam_3d_body.batch_size.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frame rate written into the side-by-side mp4.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap Stage A frame extraction to the first N frames "
                        "of the video. Keeps long clips comparable with "
                        "adiTest's 188-frame baseline. Forwarded to Stage A "
                        "only; ignored when --skip-stage-a is set (in which "
                        "case the existing intermediates dictate the cap).")
    p.add_argument("--skip-stage-a", action="store_true")
    p.add_argument("--skip-phmr", action="store_true")
    p.add_argument(
        "--skip-phmr-render", action="store_true",
        help="Skip only the per-frame PHMR mesh-overlay render. Useful "
             "when iterating on Stage D — leaves masks/run/project intact "
             "so metrics still update.",
    )
    p.add_argument("--skip-body4d", action="store_true")
    p.add_argument(
        "--skip-body4d-render", action="store_true",
        help="Skip only the per-frame body4d mesh-overlay render. Useful "
             "when iterating on Stage D — leaves the body4d run intact "
             "so metrics still update.",
    )
    p.add_argument("--skip-compare", action="store_true")
    p.add_argument(
        "--output-root", type=Path, default=None,
        help="Override cfg.output_root. Useful on the Lambda box where "
             "intermediates live under ~/work/3d_compare/<clip>/ rather "
             "than <repo>/runs/3d_compare/<clip>/.",
    )
    p.add_argument(
        "--smplx-path", type=Path, default=None,
        help="Directory with SMPLX_NEUTRAL.npz. Defaults to "
             "<phmr-repo>/data/body_models/smplx (only used by phmr_render).",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = default_config(repo_root=REPO_ROOT)
    if args.output_root is not None:
        from dataclasses import replace
        cfg = replace(cfg, output_root=args.output_root.expanduser().resolve())
    dirs = cfg.clip_dirs(args.clip).ensure()

    plan = plan_pipeline(
        skip_stage_a=args.skip_stage_a,
        skip_phmr=args.skip_phmr,
        skip_body4d=args.skip_body4d,
        skip_compare=args.skip_compare,
        skip_phmr_render=args.skip_phmr_render,
        skip_body4d_render=args.skip_body4d_render,
    )
    log.info("[%s] pipeline plan: %s", args.clip, " -> ".join(plan))

    if "stage_a" in plan:
        if args.video is None or args.cache_dir is None:
            log.error("--video and --cache-dir are required unless --skip-stage-a")
            return 64
        rc = _local_run(
            build_stage_a_cmd(
                python=Path(sys.executable),
                clip=args.clip,
                video=args.video,
                cache_dir=args.cache_dir,
                max_frames=args.max_frames,
                output_root=args.output_root,
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("stage A failed (%d)", rc)
            return rc

    if "phmr_masks" in plan:
        rc = _conda_run(
            cfg.phmr_conda_env,
            build_phmr_masks_cmd(
                intermediates_dir=dirs.intermediates,
                prompthmr_path=cfg.phmr_repo,
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("phmr_masks failed (%d)", rc)
            return rc

    if "phmr_run" in plan:
        rc = _conda_run(
            cfg.phmr_conda_env,
            build_phmr_run_cmd(
                intermediates_dir=dirs.intermediates,
                output_dir=dirs.prompthmr,
                prompthmr_path=cfg.phmr_repo,
                static_camera=args.static_camera,
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("phmr_run failed (%d)", rc)
            return rc

    if "phmr_project" in plan:
        rc = _local_run(
            build_phmr_project_joints_cmd(prompthmr_dir=dirs.prompthmr),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("phmr_project failed (%d)", rc)
            return rc

    if "phmr_render" in plan:
        smplx_path = args.smplx_path or (cfg.phmr_repo / "data" / "body_models" / "smplx")
        rc = _conda_run(
            cfg.phmr_conda_env,
            build_phmr_render_overlay_cmd(
                prompthmr_dir=dirs.prompthmr,
                frames_dir=dirs.intermediates / "frames",
                smplx_path=smplx_path,
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("phmr_render failed (%d)", rc)
            return rc

    if "body4d" in plan:
        rc = _conda_run(
            cfg.body4d_conda_env,
            build_body4d_cmd(
                intermediates_dir=dirs.intermediates,
                output_dir=dirs.sam_body4d,
                sam_body4d_path=cfg.body4d_repo,
                disable_completion=args.disable_completion,
                batch_size=args.batch_size,
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("body4d failed (%d)", rc)
            return rc

    if "body4d_render" in plan:
        rc = _conda_run(
            cfg.body4d_conda_env,
            build_body4d_render_overlay_cmd(
                body4d_dir=dirs.sam_body4d,
                frames_dir=dirs.intermediates / "frames_full",
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("body4d_render failed (%d)", rc)
            return rc

    if "compare" in plan:
        rc = _local_run(
            build_compare_cmd(
                prompthmr_joints=dirs.prompthmr / "joints_coco17_cam.npy",
                body4d_joints=dirs.sam_body4d / "joints_world.npy",
                output=dirs.comparison / "metrics.json",
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("compare failed (%d)", rc)
            return rc

    if "render" in plan:
        phmr_frames = dirs.prompthmr / "rendered_frames"
        phmr_frames_arg: Optional[Path] = phmr_frames if phmr_frames.is_dir() else None
        body4d_overlay = dirs.sam_body4d / "rendered_frames_overlay"
        body4d_frames_dir = (
            body4d_overlay if body4d_overlay.is_dir()
            else dirs.sam_body4d / "rendered_frames"
        )
        rc = _local_run(
            build_render_cmd(
                body4d_frames_dir=body4d_frames_dir,
                prompthmr_frames_dir=phmr_frames_arg,
                output=dirs.comparison / "side_by_side.mp4",
                fps=args.fps,
            ),
            cwd=REPO_ROOT,
        )
        if rc != 0:
            log.error("render failed (%d)", rc)
            return rc

    log.info(
        "[%s] all stages complete; results under %s",
        args.clip, dirs.intermediates.parent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
