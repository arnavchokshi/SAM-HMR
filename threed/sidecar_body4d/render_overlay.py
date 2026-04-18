"""SAM-Body4D mesh-overlay renderer — closes the right-panel "clean
background" gap so the side-by-side video shows both pipelines'
meshes overlaid on the original input frames.

Reads the per-dancer per-frame PLY meshes + focal JSONs that
SAM-Body4D writes (``mesh_4d_individual/<tid>/<frame:08d>.ply`` +
``focal_4d_individual/<tid>/<frame:08d>.json``) and produces one JPG
per input frame with all dancers' meshes alpha-blended onto the
original RGB image. Output goes to
``<body4d_dir>/rendered_frames_overlay/<frame:08d>.jpg``, which the
side-by-side renderer (:mod:`threed.compare.render`) prefers over
upstream's ``rendered_frames/`` whenever it's on disk.

Why a separate sidecar (and not patching SAM-Body4D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SAM-Body4D's :class:`Renderer` already does the right thing for
multi-dancer rendering: :func:`Renderer.render_rgba_multiple` adds
each dancer's mesh to a single :class:`pyrender.Scene` with the
camera at the world origin and **no** per-dancer translation —
upstream's :func:`Renderer.vertices_to_trimesh` already bakes the
per-dancer ``cam_t`` and the OpenCV→OpenGL 180° X-rotation into
the PLY-on-disk vertices. But upstream's :func:`save_mesh_results`
callsite renders onto a clean white image, so the
``rendered_frames/`` JPGs don't overlay the meshes on the dance
footage. We replicate ``render_rgba_multiple``'s scene setup
(single scene, camera at origin, per-frame intrinsics from the
JSON sidecar), then alpha-blend the resulting RGBA onto the input
frame. The structure mirrors
:mod:`threed.sidecar_promthmr.render_overlay`, so the two sidecars
share the colour palette + composite helpers.

Coordinate convention (gotcha — keep this short and pinned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PLY vertices on disk are ``(pred_vertices + cam_t) * [1, -1, -1]``.
This places the mesh's centroid at ``(cam_t.x, -cam_t.y, -cam_t.z)``
in OpenGL world coords (Y-up, Z-back), which is in front of an
origin camera looking down ``-Z``. **No further translation is
needed** in the multi-dancer scene; an earlier draft of this module
applied a ``-camera_world`` shift on top, which double-translated
each dancer to ``2*cam_t`` depth and made every mesh render at
half its true on-screen size. The pure helper
:func:`upstream_ply_centroid` documents and pins this convention so
the regression can't sneak back. See box receipt in
``runs/3d_compare/_agent_log.md`` for the visual evidence.

Inputs (all paths absolute or relative to CWD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``<body4d_dir>/mesh_4d_individual/<tid>/<frame:08d>.ply`` —
  per-dancer per-frame mesh emitted by SAM-Body4D.
- ``<body4d_dir>/focal_4d_individual/<tid>/<frame:08d>.json`` —
  ``{"focal_length": float, "camera": [x, y, z]}`` written by
  SAM-Body4D's :func:`save_mesh_results`.
- ``<frames_dir>/<frame:08d>.jpg`` — original RGB frames, e.g.
  Stage A's ``intermediates/frames_full/`` at the resolution
  SAM-Body4D ran on.

Output
~~~~~~

- ``<body4d_dir>/rendered_frames_overlay/<frame:08d>.jpg`` — BGR
  JPGs at the input resolution. Frames where no dancer has a PLY
  on disk are written as a copy of the input image to keep frame
  counts in sync with the upstream ``rendered_frames/`` for the
  side-by-side stitcher.

Pure helpers (unit-tested in
``tests/threed/test_body4d_render_overlay.py``):

- :func:`discover_body4d_dancer_ids`
- :func:`load_focal_meta`
- :func:`upstream_ply_centroid`
- :func:`flip_yz_verts`

The pyrender rasterisation loop in :func:`main` is exercised by the
box smoke (logged in ``runs/3d_compare/_agent_log.md``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from threed.sidecar_promthmr.render_overlay import (
    DEFAULT_MESH_ALPHA,
    composite_overlay,
    dancer_color_palette,
    make_intrinsics_K,
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested locally; no torch / pyrender / trimesh import here).
# ---------------------------------------------------------------------------


def discover_body4d_dancer_ids(mesh_root: Path) -> List[int]:
    """Return sorted dancer ids found as numeric subdirs under ``mesh_root``.

    SAM-Body4D writes ``mesh_4d_individual/<tid>/`` per dancer (tid
    is an int like ``1``, ``2``, ...). We skip non-digit subdirs so
    a stray cache directory doesn't crash the renderer, and return
    ``[]`` if the root is missing entirely so callers can no-op
    instead of raising on a partial run.
    """
    if not mesh_root.is_dir():
        return []
    out: List[int] = []
    for p in mesh_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append(int(p.name))
    return sorted(out)


def load_focal_meta(json_path: Path) -> Tuple[float, np.ndarray]:
    """Return ``(focal_length, camera_xyz)`` from a SAM-Body4D focal JSON.

    The JSON shape is ``{"focal_length": float, "camera": [x, y, z]}``;
    see ``save_mesh_results`` in
    ``models/sam_3d_body/notebook/utils.py``. Focal is a per-dancer
    per-frame scalar (in pixels at the body4d resolution). Camera is
    the upstream ``pred_cam_t`` translation in metric model space.
    """
    with open(json_path) as f:
        meta = json.load(f)
    return float(meta["focal_length"]), np.asarray(meta["camera"], dtype=np.float32)


def upstream_ply_centroid(cam_t: np.ndarray) -> np.ndarray:
    """Approximate centroid of a SAM-Body4D PLY-on-disk in world coords.

    Models SAM-Body4D's :func:`Renderer.vertices_to_trimesh` for a
    centred mesh (``pred_vertices`` ≈ origin in model space): the
    saved PLY vertices are ``(pred_vertices + cam_t) * [1, -1, -1]``,
    so the centroid sits at ``(cam_t.x, -cam_t.y, -cam_t.z)`` in
    OpenGL world coords. For typical positive ``cam_t.z`` this is in
    front of an origin camera looking down ``-Z``, which is why
    :func:`Renderer.render_rgba_multiple` (and our
    :func:`_build_pyrender_scene`) needs **no per-dancer translation**.

    Used purely as a regression pin: a future change that re-introduces
    a ``-camera_world`` offset on top of the PLY would fail
    :class:`tests.threed.test_body4d_render_overlay.TestUpstreamPlyCentroid`
    long before reaching the box. See module docstring for the bug it
    guards against.
    """
    cam_t = np.asarray(cam_t, dtype=np.float32)
    if cam_t.shape != (3,):
        raise ValueError(f"cam_t must be (3,), got {cam_t.shape}")
    return np.array(
        [cam_t[0], -cam_t[1], -cam_t[2]],
        dtype=np.float32,
    )


def flip_yz_verts(verts: np.ndarray) -> np.ndarray:
    """Negate Y/Z of vertices — equivalent to a 180° rotation around X.

    Generic geometry helper. Not used by the body4d overlay path
    (PLYs come out of SAM-Body4D's :func:`vertices_to_trimesh`
    already 180°-X-rotated so they're consumed as-is), but kept as a
    standalone utility because the same convention conversion is
    common when bridging OpenCV-style and OpenGL-style coordinates.
    Mirrors ``trimesh.transformations.rotation_matrix(np.radians(180),
    [1, 0, 0])`` without requiring trimesh.
    """
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (N, 3); got {verts.shape}")
    out = verts.astype(np.float32, copy=True)
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


# ---------------------------------------------------------------------------
# GPU-only entry point (trimesh + pyrender; box-side).
# ---------------------------------------------------------------------------


def _build_pyrender_scene(
    meshes_with_cam: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float]]],
    *,
    focal: float,
    img_h: int,
    img_w: int,
):
    """Build a pyrender Scene with all dancers + a perspective camera.

    Mirrors :func:`Renderer.render_rgba_multiple` in
    ``models/sam_3d_body/sam_3d_body/visualization/renderer.py``:
    one scene, camera at the world origin, per-dancer mesh added
    **as-is** from the PLY. The PLY already encodes the per-dancer
    ``cam_t`` translation and the OpenCV→OpenGL X-rotation via
    upstream's :func:`vertices_to_trimesh`, so adding any extra
    per-dancer offset here would double-translate the mesh and
    shrink it on screen (see the "Coordinate convention" gotcha in
    the module docstring).

    Each entry in ``meshes_with_cam`` is a tuple
    ``(verts, faces, cam_t, color)``:

    - ``verts``: ``(V, 3)`` mesh vertices straight off the PLY,
      consumed verbatim.
    - ``faces``: ``(F, 3)`` mesh faces.
    - ``cam_t``: ``(3,)`` per-dancer ``pred_cam_t``. Currently unused
      by the scene math — kept in the tuple because the loader has it
      for free and a future mode (e.g. per-dancer focal compensation)
      may want it.
    - ``color``: RGB triple in ``[0, 1]``.

    The camera intrinsics use the shared per-frame focal +
    image-centred principal point (``[W/2, H/2]``) — same convention
    :func:`render_rgba_multiple` uses when ``camera_center`` is left
    as the default.
    """
    import pyrender
    import trimesh

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    for verts, faces, _cam_t, color in meshes_with_cam:
        v = verts.astype(np.float32, copy=True)
        mesh_tri = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[float(color[0]), float(color[1]), float(color[2]), 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.7,
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh_tri, material=material, smooth=True))

    K = make_intrinsics_K(focal=float(focal), cx=float(img_w) / 2.0, cy=float(img_h) / 2.0)
    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        znear=0.1, zfar=1e6,
    )
    scene.add(camera, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=np.eye(4))
    return scene


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--body4d-dir", type=Path, required=True,
        help="Directory with mesh_4d_individual/, focal_4d_individual/, etc. "
             "(Stage C2 output).",
    )
    p.add_argument(
        "--frames-dir", type=Path, required=True,
        help="Directory of original input JPGs at body4d's resolution "
             "(typically <intermediates>/frames_full/).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir (defaults to <body4d-dir>/rendered_frames_overlay/).",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_MESH_ALPHA,
        help=f"Per-mesh blending alpha (default {DEFAULT_MESH_ALPHA}).",
    )
    args = p.parse_args(argv)

    body4d_dir: Path = args.body4d_dir.expanduser().resolve()
    frames_dir: Path = args.frames_dir.expanduser().resolve()
    out_dir: Path = (args.output_dir or body4d_dir / "rendered_frames_overlay").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_root = body4d_dir / "mesh_4d_individual"
    focal_root = body4d_dir / "focal_4d_individual"
    dancer_ids = discover_body4d_dancer_ids(mesh_root)
    if not dancer_ids:
        print(f"[body4d_overlay] ERROR: no dancers under {mesh_root}", file=sys.stderr)
        return 2

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[body4d_overlay] ERROR: no JPG frames in {frames_dir}", file=sys.stderr)
        return 2

    palette = dancer_color_palette(len(dancer_ids))
    print(f"[body4d_overlay] {len(frame_paths)} frames, dancers={dancer_ids}")

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import pyrender
    import trimesh

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        print(f"[body4d_overlay] ERROR: unreadable {frame_paths[0]}", file=sys.stderr)
        return 2
    H, W = sample.shape[:2]
    print(f"[body4d_overlay] frame size {W}x{H}, alpha={args.alpha}")

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    n_drawn = 0
    n_pass_through = 0
    try:
        for ti, fp in enumerate(frame_paths):
            frame_stem = fp.stem
            img_bgr = cv2.imread(str(fp))
            if img_bgr is None:
                print(f"[body4d_overlay] WARN: unreadable {fp}, skipping")
                continue

            meshes_with_cam: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            focal_consensus: Optional[float] = None
            for di, dancer_id in enumerate(dancer_ids):
                ply_path = mesh_root / str(dancer_id) / f"{frame_stem}.ply"
                json_path = focal_root / str(dancer_id) / f"{frame_stem}.json"
                if not ply_path.is_file() or not json_path.is_file():
                    continue
                focal, cam = load_focal_meta(json_path)
                if focal_consensus is None:
                    focal_consensus = focal
                mesh_t = trimesh.load(str(ply_path), process=False)
                verts = np.asarray(mesh_t.vertices, dtype=np.float32)
                faces = np.asarray(mesh_t.faces, dtype=np.int32)
                meshes_with_cam.append((verts, faces, cam, palette[di]))

            if not meshes_with_cam or focal_consensus is None:
                cv2.imwrite(str(out_dir / fp.name), img_bgr)
                n_pass_through += 1
                continue

            scene = _build_pyrender_scene(
                meshes_with_cam,
                focal=focal_consensus,
                img_h=H,
                img_w=W,
            )
            color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = np.asarray(color)
            rgb_bgr = color[..., :3][..., ::-1]
            alpha = color[..., 3].astype(np.float32) / 255.0
            alpha = np.clip(alpha * args.alpha, 0.0, 1.0)
            blended = composite_overlay(img_bgr, rgb_bgr, alpha)
            cv2.imwrite(str(out_dir / fp.name), blended)
            n_drawn += 1

            if (ti + 1) % 20 == 0:
                print(
                    f"[body4d_overlay] {ti + 1}/{len(frame_paths)} "
                    f"({n_drawn} drawn, {n_pass_through} pass-through)"
                )
    finally:
        renderer.delete()

    print(
        f"[body4d_overlay] wrote {len(frame_paths)} JPGs to {out_dir} "
        f"({n_drawn} with meshes, {n_pass_through} pass-through)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
