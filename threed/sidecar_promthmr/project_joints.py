"""PromptHMR-side joint projector — Stage D pre-processing (plan Task 11f).

Reads PromptHMR-Vid's per-clip outputs and writes a COCO-17, cam-frame
joints file for the comparison driver:

Inputs:
- ``<prompthmr_dir>/results.pkl``    — full Pipeline.results dict; we
  read ``camera_world.Rcw`` (T, 3, 3) and ``camera_world.Tcw`` (T, 3).
- ``<prompthmr_dir>/joints_world.npy`` — (T, N_dancers, 22, 3) SMPL
  body joints in world frame, written by Task 7's runner.

Output:
- ``<output>``                       — (T, N_dancers, 17, 3) COCO-17
  joints in cam frame, ready for ``threed.compare.run_compare``.

The world->cam projection is per-frame (PromptHMR's SLAM provides
moving-camera extrinsics; for ``--static-camera`` clips Rcw is identity
and Tcw is constant, so the projection is a no-op shift). NaN frames
are preserved so a partially-tracked dancer stays NaN downstream.

The SMPL-22 -> COCO-17 reduction uses :data:`SMPL22_TO_COCO17`
(face indices collapse to SMPL ``head`` — see
``threed.compare.joints`` docstring). MPJPE on COCO 0..4 will therefore
be uninformative; report the per-joint values and let the consumer
filter the face indices out as needed.

Pure-host module — no PromptHMR or torch import. Runs in the
``threed-host`` env and can be re-run independently of GPU work
(useful when iterating on Stage D without re-running the heavy
PromptHMR pipeline).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from threed.compare.joints import smpl22_to_coco17


def project_joints_world_to_cam(
    joints_world: np.ndarray,
    Rcw: np.ndarray,
    Tcw: np.ndarray,
) -> np.ndarray:
    """Project ``(T, N, J, 3)`` world joints into per-frame cam coords.

    For each frame ``t``::

        joints_cam[t] = (Rcw[t] @ joints_world[t].reshape(-1, 3).T).T + Tcw[t]

    Implemented as a single ``np.einsum`` for vectorised efficiency.
    NaN entries propagate (PromptHMR fills NaN for absent dancers
    in :func:`threed.sidecar_promthmr.run_promthmr_vid.joints_world_padded`).

    Parameters
    ----------
    joints_world : np.ndarray
        ``(T, N, J, 3)`` world joints in metres.
    Rcw : np.ndarray
        ``(T, 3, 3)`` rotation matrices (camera-from-world).
    Tcw : np.ndarray
        ``(T, 3)`` translations (camera-from-world).

    Returns
    -------
    np.ndarray
        ``(T, N, J, 3)`` joints in the per-frame cam coordinates.

    Raises
    ------
    ValueError
        If the frame counts of ``joints_world``, ``Rcw``, ``Tcw`` disagree.
    """
    T = joints_world.shape[0]
    if Rcw.shape[0] != T or Tcw.shape[0] != T:
        raise ValueError(
            f"frame count mismatch: joints={T}, Rcw={Rcw.shape[0]}, Tcw={Tcw.shape[0]}"
        )
    rotated = np.einsum("tij,tnkj->tnki", Rcw, joints_world)
    return rotated + Tcw[:, None, None, :]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompthmr-dir", type=Path, required=True,
        help="Directory written by Stage C1 — must contain results.pkl + joints_world.npy.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the (T, N, 17, 3) cam-frame COCO-17 npy.",
    )
    p.add_argument(
        "--no-projection", action="store_true",
        help="Skip the world->cam projection (just reduce SMPL-22 -> COCO-17). "
             "Useful for static-camera tripod clips where world == cam.",
    )
    args = p.parse_args(argv)

    pdir: Path = args.prompthmr_dir.expanduser().resolve()
    joints_path = pdir / "joints_world.npy"
    results_path = pdir / "results.pkl"
    if not joints_path.is_file():
        print(f"[project_joints] ERROR: missing {joints_path}", file=sys.stderr)
        return 2
    if not results_path.is_file() and not args.no_projection:
        print(f"[project_joints] ERROR: missing {results_path}", file=sys.stderr)
        return 2

    joints_world = np.load(joints_path)
    print(f"[project_joints] joints_world {joints_world.shape}, dtype={joints_world.dtype}")

    if args.no_projection:
        joints_cam = joints_world
        print("[project_joints] --no-projection: treating world == cam")
    else:
        import joblib
        results = joblib.load(results_path)
        camw = results["camera_world"]
        Rcw = np.asarray(camw["Rcw"], dtype=np.float32)
        Tcw = np.asarray(camw["Tcw"], dtype=np.float32)
        if Tcw.ndim == 3 and Tcw.shape[-1] == 1:
            Tcw = Tcw[..., 0]
        print(f"[project_joints] Rcw {Rcw.shape}, Tcw {Tcw.shape}")
        joints_cam = project_joints_world_to_cam(joints_world, Rcw, Tcw)

    coco17 = smpl22_to_coco17(joints_cam).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, coco17)
    print(f"[project_joints] wrote {args.output} {coco17.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
