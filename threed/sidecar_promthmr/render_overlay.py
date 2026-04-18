"""PromptHMR-Vid mesh-overlay renderer — closes Stage D's open followup #3.

Reads PromptHMR-Vid's cached ``results.pkl`` and writes one JPG per
input frame with all dancers' SMPL-X meshes overlaid on the original
RGB image. The output directory ``<phmr_dir>/rendered_frames/`` is the
exact path Stage D's :mod:`threed.compare.render` consumes via
``--prompthmr-frames-dir`` — when this script has been run, the
side-by-side video shows real PHMR meshes instead of a black panel.

Design — why a separate script (not part of run_promthmr_vid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PromptHMR-Vid's GPU path takes ~5–8 min per clip (DROID-SLAM +
ViTPose + SLAM refinement + post-opt). The render is independent —
once ``results.pkl`` is on disk we can re-run rendering as many times
as we like (tweak colors, alpha, framing) without paying the GPU cost
again. Keeping it separate also lets the orchestrator skip rendering
selectively (``--skip-phmr-render``) when iterating on Stage D.

Inputs (all paths absolute or relative to CWD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``<prompthmr_dir>/results.pkl`` — Pipeline.results dict with
  ``people[tid].smplx_cam`` (rotmat, shape, trans, frames) and
  ``camera_world.img_focal``, ``camera_world.img_center``.
- ``<frames_dir>/<frame:08d>.jpg`` — original RGB frames (e.g.
  Stage A's ``intermediates/frames/`` at the same resolution PHMR was
  fed).
- ``<smplx_path>`` — directory containing ``SMPLX_NEUTRAL.npz``
  (defaults to ``$PROMPTHMR_PATH/data/body_models/smplx``).

Output
~~~~~~

- ``<prompthmr_dir>/rendered_frames/<frame:08d>.jpg`` — BGR JPGs
  at the input resolution. Frames where no dancer is detected are
  written as a copy of the input image (to keep frame counts in sync
  with Stage C2's body4d renders).

Pure helpers (unit-tested in
``tests/threed/test_promthmr_render_overlay.py``):

- :data:`DEFAULT_MESH_ALPHA`
- :func:`dancer_color_palette`
- :func:`pose_axis_angle_from_rotmat`
- :func:`make_intrinsics_K`
- :func:`frame_dancer_index`
- :func:`composite_overlay`

The GPU rasterisation loop in :func:`main` is exercised by the box
smoke (Plan Task 11g extension; logged in ``runs/3d_compare/_agent_log.md``).
"""
from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Mesh fragments are alpha-blended onto the input image with this
# constant so the underlying body is still visible. Pinned in [0, 1] by
# :class:`tests.threed.test_promthmr_render_overlay.TestCompositeOverlay`.
DEFAULT_MESH_ALPHA: float = 0.65


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested locally; no torch / smplx / pyrender import here).
# ---------------------------------------------------------------------------


def dancer_color_palette(n: int) -> np.ndarray:
    """Return ``n`` distinct RGB colors in ``(n, 3)`` float32, range [0, 1].

    Walks the HSV color wheel at full saturation/value so adjacent
    dancers always have visually distinct hues. Deterministic: the
    same ``n`` always returns the same palette so re-renders don't
    flicker between runs.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out[i] = (r, g, b)
    return out


def pose_axis_angle_from_rotmat(rotmat: np.ndarray) -> np.ndarray:
    """Convert ``(B, J, 3, 3)`` rotation matrices to ``(B, J*3)`` axis-angle.

    Uses scipy's Rodrigues conversion (``Rotation.from_matrix``) for
    numerical robustness; the result is float32 so it can be fed
    straight into ``smplx.SMPLX.forward(..., pose2rot=False)`` after a
    reshape (the smplx package accepts either format, but axis-angle
    is the more natural intermediate representation when the on-disk
    artefact is a rotation matrix).

    Mirrors the ``matrix_to_axis_angle(...).reshape(-1, 55*3)`` line in
    PromptHMR's :func:`pipeline.world.world_hps_estimation`.
    """
    if rotmat.ndim != 4 or rotmat.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotmat must be (B, J, 3, 3); got {rotmat.shape}"
        )
    from scipy.spatial.transform import Rotation as Rsp

    B, J = rotmat.shape[:2]
    flat = rotmat.reshape(-1, 3, 3)
    rotvec = Rsp.from_matrix(flat).as_rotvec().astype(np.float32)
    return rotvec.reshape(B, J * 3)


def make_intrinsics_K(focal: float, cx: float, cy: float) -> np.ndarray:
    """Build a ``(3, 3)`` pinhole intrinsics matrix in float32."""
    K = np.array([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K


def frame_dancer_index(
    per_dancer_frames: Sequence[np.ndarray],
    n_frames: int,
) -> List[List[Tuple[int, int]]]:
    """Group per-dancer frame lists into a per-frame ``(dancer_idx, local_idx)`` list.

    Returned list has length ``n_frames``; entry ``t`` is a list of
    ``(dancer_idx, local_idx)`` pairs telling the renderer which
    dancers are present at frame ``t`` and which row of their SMPL-X
    arrays to read. Out-of-range frames (>= ``n_frames``) are dropped
    silently (PHMR may pad track lists).
    """
    out: List[List[Tuple[int, int]]] = [[] for _ in range(n_frames)]
    for di, frames in enumerate(per_dancer_frames):
        for li, t in enumerate(np.asarray(frames, dtype=np.int64)):
            if 0 <= int(t) < n_frames:
                out[int(t)].append((di, li))
    return out


def composite_overlay(
    rgb_input: np.ndarray,
    render_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Alpha-blend ``render_rgb`` onto ``rgb_input`` per pixel.

    All inputs share spatial size ``(H, W)``; alpha is a 2D float
    array in ``[0, 1]`` and the colour inputs are uint8. Blending
    happens in float32 and the result is rounded back to uint8.
    """
    if rgb_input.shape != render_rgb.shape:
        raise ValueError(
            f"rgb_input shape {rgb_input.shape} != render_rgb shape {render_rgb.shape}"
        )
    if alpha.shape != rgb_input.shape[:2]:
        raise ValueError(
            f"alpha shape {alpha.shape} != HxW {rgb_input.shape[:2]}"
        )
    a = alpha[..., None].astype(np.float32)
    blended = rgb_input.astype(np.float32) * (1.0 - a) + render_rgb.astype(np.float32) * a
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GPU-only entry point (smplx + pyrender; box-side).
# ---------------------------------------------------------------------------


def _smplx_verts_per_dancer(
    smplx_model,
    smplx_cam: Dict,
    *,
    device: str,
) -> np.ndarray:
    """Run smplx forward for one dancer; return ``(T, V, 3)`` verts in cam frame.

    Mirrors the working pattern in
    ``threed.sidecar_promthmr.run_promthmr_vid._extract_smplx_body_joints_world``
    (and ``pipeline.world.world_hps_estimation`` upstream) — convert
    the on-disk rotation matrices to axis-angle (165 = 55*3) and feed
    smplx in the axis-angle path, with zeros for jaw/leye/reye/expression
    (face joints don't change the body mesh appreciably).

    We ran into a smplx ``pose2rot=False`` shape-validation error trying
    to feed (B, J, 3, 3) rotation matrices straight in, so the
    axis-angle round-trip is the safer of the two paths and matches the
    code we already have in production.
    """
    import torch

    rotmat = np.asarray(smplx_cam["rotmat"], dtype=np.float32)
    pose_aa_np = pose_axis_angle_from_rotmat(rotmat)
    pose_aa = torch.as_tensor(pose_aa_np, dtype=torch.float32, device=device)
    shape = torch.as_tensor(smplx_cam["shape"], dtype=torch.float32, device=device)
    trans = torch.as_tensor(smplx_cam["trans"], dtype=torch.float32, device=device)
    B = pose_aa.shape[0]
    if shape.shape[1] > smplx_model.num_betas:
        shape = shape[:, : smplx_model.num_betas]
    mean_shape = shape.mean(dim=0, keepdim=True).repeat(B, 1)
    zeros3 = torch.zeros(B, 3, dtype=pose_aa.dtype, device=device)
    with torch.no_grad():
        out = smplx_model(
            global_orient=pose_aa[:, 0:3],
            body_pose=pose_aa[:, 3:66],
            left_hand_pose=pose_aa[:, 75:120],
            right_hand_pose=pose_aa[:, 120:165],
            betas=mean_shape,
            transl=trans,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=torch.zeros(B, 10, dtype=pose_aa.dtype, device=device),
        )
    return out.vertices.detach().cpu().numpy().astype(np.float32)


def _build_pyrender_scene(
    verts_per_dancer: List[np.ndarray],
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    K: np.ndarray,
    img_h: int,
    img_w: int,
):
    """Build a pyrender Scene with one mesh per dancer + a perspective camera.

    Camera convention follows pyrender (OpenGL-ish, looks down -z, y up).
    PHMR's `smplx_cam` is in OpenCV convention (looks down +z, y down),
    so the meshes are flipped in y/z before being added — same trick
    PromptHMR's pytorch3d renderer uses internally
    (``rot_180 = diag(1, -1, -1)`` in ``demo_phmr.py:73-78``).
    """
    import pyrender
    import trimesh

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    flip_yz = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    for verts, color in zip(verts_per_dancer, colors):
        v = verts @ flip_yz.T
        mesh_tri = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[float(color[0]), float(color[1]), float(color[2]), 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.7,
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh_tri, material=material, smooth=True))

    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        znear=0.1, zfar=100.0,
    )
    scene.add(camera, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=np.eye(4))
    return scene


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompthmr-dir", type=Path, required=True,
        help="Directory with results.pkl (Stage C1 output).",
    )
    p.add_argument(
        "--frames-dir", type=Path, required=True,
        help="Directory of original input JPGs (e.g. intermediates/frames).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir for rendered_frames/. Defaults to "
             "<prompthmr-dir>/rendered_frames/.",
    )
    p.add_argument(
        "--smplx-path", type=Path,
        default=Path(os.environ.get(
            "SMPLX_PATH",
            "~/code/PromptHMR/data/body_models/smplx",
        )).expanduser(),
        help="Directory containing SMPLX_NEUTRAL.npz.",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_MESH_ALPHA,
        help=f"Per-mesh blending alpha (default {DEFAULT_MESH_ALPHA}).",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for the SMPL-X forward (default cuda).",
    )
    args = p.parse_args(argv)

    pdir: Path = args.prompthmr_dir.expanduser().resolve()
    frames_dir: Path = args.frames_dir.expanduser().resolve()
    out_dir: Path = (args.output_dir or pdir / "rendered_frames").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_pkl = pdir / "results.pkl"
    if not results_pkl.is_file():
        print(f"[render_overlay] ERROR: missing {results_pkl}", file=sys.stderr)
        return 2

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[render_overlay] ERROR: no JPG frames in {frames_dir}", file=sys.stderr)
        return 2

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import joblib
    import smplx as smplx_pkg
    import pyrender

    print(f"[render_overlay] loading {results_pkl}")
    results = joblib.load(results_pkl)
    camw = results["camera_world"]
    focal = float(np.asarray(camw["img_focal"]).reshape(-1)[0])
    img_center = np.asarray(camw["img_center"]).reshape(-1)
    cx, cy = float(img_center[0]), float(img_center[1])

    sample = cv2.imread(str(frame_paths[0]))
    H, W = sample.shape[:2]
    K = make_intrinsics_K(focal, cx, cy)

    print(f"[render_overlay] {len(frame_paths)} frames, {W}x{H}, focal={focal}, center=({cx:.1f},{cy:.1f})")

    smplx_model = smplx_pkg.SMPLX(
        str(args.smplx_path),
        gender="neutral",
        num_betas=10,
        use_pca=False,
        use_face_contour=False,
    ).to(args.device)
    faces = smplx_model.faces.astype(np.int32)

    people = results["people"]
    tids = sorted(int(k) for k in people.keys())
    print(f"[render_overlay] dancers={tids}")
    verts_per_dancer: List[np.ndarray] = []
    frames_per_dancer: List[np.ndarray] = []
    for tid in tids:
        v = _smplx_verts_per_dancer(smplx_model, people[tid]["smplx_cam"], device=args.device)
        verts_per_dancer.append(v)
        frames_per_dancer.append(np.asarray(people[tid]["frames"], dtype=np.int64))
    palette = dancer_color_palette(len(tids))

    n_frames = len(frame_paths)
    fdi = frame_dancer_index(frames_per_dancer, n_frames=n_frames)

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    try:
        for ti, fp in enumerate(frame_paths):
            img_bgr = cv2.imread(str(fp))
            present = fdi[ti]
            if not present:
                cv2.imwrite(str(out_dir / fp.name), img_bgr)
                continue
            verts_list = [verts_per_dancer[di][li] for (di, li) in present]
            colors = palette[[di for (di, _) in present]]
            scene = _build_pyrender_scene(
                verts_list, faces, colors,
                K=K, img_h=H, img_w=W,
            )
            color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = np.asarray(color)
            rgb = color[..., :3][..., ::-1]
            alpha = color[..., 3].astype(np.float32) / 255.0
            alpha = np.clip(alpha * args.alpha, 0.0, 1.0)
            blended = composite_overlay(img_bgr, rgb, alpha)
            cv2.imwrite(str(out_dir / fp.name), blended)
            if (ti + 1) % 20 == 0:
                print(f"[render_overlay] {ti + 1}/{n_frames} frames")
    finally:
        renderer.delete()

    print(f"[render_overlay] wrote {n_frames} JPGs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
