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

SAM-Body4D's :class:`Renderer` already knows the right convention
(it negates ``cam_t.x`` for the camera position and applies a 180°
X-axis rotation to the mesh so it lands in the OpenGL frame). But
its :func:`save_mesh_results` callsite uses a clean white image as
the background, so the upstream ``rendered_frames/`` JPGs don't
overlay the meshes on the dance footage. We re-use the convention
(see :func:`body4d_dancer_world_pos` + :func:`flip_yz_verts`) but
render all dancers in **one** pyrender scene so cross-dancer
occlusion is depth-correct, then alpha-blend onto the input frame.
This also matches the structure of
:mod:`threed.sidecar_promthmr.render_overlay`, so the two sidecars
share the colour palette + composite helpers.

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
- :func:`body4d_dancer_world_pos`
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


def body4d_dancer_world_pos(cam_t: np.ndarray) -> np.ndarray:
    """Translation to apply to a PLY's vertices for the shared-camera scene.

    SAM-Body4D's :class:`Renderer` puts the **camera** at
    ``(-cam_t.x, cam_t.y, cam_t.z)`` and consumes the PLY vertices
    as-is (they were already saved in that "post-flip world" frame by
    upstream's :func:`vertices_to_trimesh` —
    ``(pred_verts + cam_t) * [1, -1, -1]``).

    For our multi-dancer scene we want the camera at the world origin
    so all dancers can share one pyrender camera (depth-correct
    cross-dancer occlusion). To put a dancer in that shared frame we
    translate their PLY vertices by ``-camera_world``, i.e.
    ``+(cam_t.x, -cam_t.y, -cam_t.z)``. The X sign comes out positive
    because upstream first negates ``cam_t.x`` to get the camera
    position; our ``-camera_world`` flips it back to ``+cam_t.x``.

    Returns the per-dancer translation vector; callers add it to the
    PLY vertices before adding the mesh to the scene.
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

    Each entry in ``meshes_with_cam`` is a tuple
    ``(verts, faces, cam_t, color)``:

    - ``verts``: ``(V, 3)`` mesh vertices straight off the PLY. These
      are SAM-Body4D's ``vertices_to_trimesh`` output, i.e.
      ``(pred_vertices + cam_t) * [1, -1, -1]`` — already in the
      "post-flip" world that upstream's :class:`Renderer` consumes,
      so we **don't** flip again here.
    - ``faces``: ``(F, 3)`` mesh faces.
    - ``cam_t``: ``(3,)`` per-dancer ``pred_cam_t`` (the un-negated
      original; we apply the X-negation inside
      :func:`body4d_dancer_world_pos`).
    - ``color``: RGB triple in ``[0, 1]``.

    Vertices are translated by :func:`body4d_dancer_world_pos` so all
    dancers land in a shared frame with the camera at the world
    origin. The camera intrinsics use the shared per-frame focal +
    image-centred principal point (``[W/2, H/2]``) — same convention
    upstream's :class:`Renderer` uses when ``camera_center`` is left
    as the default.
    """
    import pyrender
    import trimesh

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    for verts, faces, cam_t, color in meshes_with_cam:
        v = verts.astype(np.float32, copy=True) + body4d_dancer_world_pos(cam_t)[None, :]
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
