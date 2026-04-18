"""PromptHMR-Vid sidecar runner — Stage C1 of the dual 3D pipeline.

Consumes the artifacts produced by Tasks 1-6 (DeepOcSort tracks +
extracted frames + SAM-2 masks) and delegates the actual mesh
recovery to PromptHMR's :class:`Pipeline` with ``cfg.tracker = 'sam2'``
so that ``hps_estimation`` routes our masks through the mask-prompt
path.

Reads from ``--intermediates-dir``::

    frames/                # JPGs at max_height=896
    tracks.pkl             # joblib dict {tid -> {frames, bboxes, confs}}
    masks_per_track/<tid>/<frame:08d>.png   # 8-bit binary
    masks_union.npy        # (T, H, W) bool
    camera/intrinsics.json # (optional)

Writes to ``--output-dir``::

    results.pkl            # PromptHMR's full ``self.results`` dict
    world4d.mcs            # MeshCapade scene (drag into me.meshcapade.com/editor)
    world4d.glb            # auto-converted from the .mcs (via export_scene_with_camera)
    subject-<tid>.smpl     # one per dancer
    joints_world.npy       # (n_frames, n_dancers, 22, 3) NaN-padded — Stage D input

The runner stays pure (no upstream PromptHMR file edits at runtime).
The one-time edit of ``PromptHMR/pipeline/phmr_vid.py:23`` to point at
``phmr_b1b2.ckpt`` is performed by the box-side wrapper script
(``~/work/run_task7_smoke.sh``) and is recorded in the agent log.

GPU-free helpers exposed for testing::

    intermediates_layout_ok, load_per_track_masks,
    sorted_tid_list, joints_world_padded
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import joblib
import numpy as np

from threed.sidecar_promthmr.build_masks import (
    chdir_to_prompthmr,
    inject_prompthmr_path,
)


# ---------------------------------------------------------------------------
# GPU-free helpers (unit-tested)
# ---------------------------------------------------------------------------


def intermediates_layout_ok(interm: Path) -> Tuple[bool, List[str]]:
    """Check the artifact layout produced by Tasks 1-6.

    Returns ``(ok, errors)`` so callers can show every missing artifact
    in one shot instead of failing on the first one. The contract is
    pinned by ``TestIntermediatesLayoutOk`` in
    ``tests/threed/test_sidecar_promthmr_run_promthmr_vid.py``.
    """
    errs: List[str] = []
    interm = Path(interm)

    frames_dir = interm / "frames"
    if not frames_dir.is_dir() or not any(frames_dir.glob("*.jpg")):
        errs.append(f"frames/ directory missing or empty under {interm}")

    if not (interm / "tracks.pkl").is_file():
        errs.append(f"tracks.pkl missing under {interm}")

    if not (interm / "masks_union.npy").is_file():
        errs.append(f"masks_union.npy missing under {interm}")

    masks_pt = interm / "masks_per_track"
    if not masks_pt.is_dir():
        errs.append(f"masks_per_track/ directory missing under {interm}")

    return len(errs) == 0, errs


def load_per_track_masks(
    interm: Path,
    tracks: Dict,
    H: int,
    W: int,
) -> Dict:
    """Mutate ``tracks`` in place, attaching ``masks``, ``track_id``, ``detected``.

    PromptHMR's ``Pipeline`` expects ``results['people'][tid]`` to expose:

    - ``masks``     : (T_track, H, W) bool — per-frame mask for the track
    - ``track_id``  : python int — used as a dict key downstream
    - ``detected``  : (T_track,) bool — which frames have valid bboxes (we
                      trust DeepOcSort and mark them all True)

    Per-frame PNGs come from ``masks_per_track/<tid>/<frame:08d>.png``.
    Missing PNGs (track ID present in ``tracks.pkl`` but mask not produced)
    fall back to an all-zero mask so PromptHMR doesn't crash; the
    ``detected`` flag stays True regardless because the track ID exists.

    Returns the same ``tracks`` dict for chaining.
    """
    interm = Path(interm)
    masks_root = interm / "masks_per_track"
    for tid, t in tracks.items():
        tid_int = int(tid)
        per_t = []
        for f in t["frames"]:
            png = masks_root / str(tid_int) / f"{int(f):08d}.png"
            if png.is_file():
                m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) > 127
            else:
                m = np.zeros((H, W), dtype=bool)
            per_t.append(m)
        t["masks"] = np.stack(per_t).astype(bool)
        t["track_id"] = tid_int
        t["detected"] = np.ones(len(t["frames"]), dtype=bool)
    return tracks


def sorted_tid_list(tracks: Dict) -> List[int]:
    """Stable, python-int sort of track IDs (joblib emits ``np.int64``)."""
    return sorted(int(k) for k in tracks.keys())


def joints_world_padded(
    per_track_joints: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_frames: int,
    tid_order: Sequence[int],
) -> np.ndarray:
    """Pack ``{tid -> (frames_idx, joints_T_22_3)}`` into ``(N, D, 22, 3)``.

    Frames where a dancer is absent are filled with ``NaN`` so that
    Stage D can compute pairwise metrics with `nanmean`/`nanstd`
    without manual masks. ``tid_order`` controls the ``D`` axis (use
    ``sorted_tid_list`` for the canonical order). Joint count is fixed
    at 22 (the SMPL body subset, indices 0:22 of SMPL-X output).
    """
    out = np.full((n_frames, len(tid_order), 22, 3), np.nan, dtype=np.float32)
    for di, tid in enumerate(tid_order):
        if tid not in per_track_joints:
            continue
        frames_idx, joints = per_track_joints[tid]
        for fi, frame in enumerate(np.asarray(frames_idx, dtype=np.int64)):
            if 0 <= int(frame) < n_frames:
                out[int(frame), di] = joints[fi]
    return out


# ---------------------------------------------------------------------------
# GPU-only entry point
# ---------------------------------------------------------------------------


def _read_optional_intrinsics(interm: Path) -> dict | None:
    p = interm / "camera" / "intrinsics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _load_frames_rgb(frames_dir: Path) -> np.ndarray:
    """Load JPGs as a stacked ``(N, H, W, 3)`` RGB uint8 numpy array.

    PromptHMR's ``load_video_frames`` returns a 4D ndarray (not a list)
    and several downstream helpers rely on that — e.g.
    ``run_spec_calib`` does ``isinstance(images, np.ndarray)`` to derive
    the image size, and a list of arrays falls through both branches
    leaving ``imgsize`` undefined (UnboundLocalError). Stacking keeps
    the contract aligned with the upstream Pipeline.
    """
    paths = sorted(frames_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no JPG frames under {frames_dir}")
    return np.stack([cv2.imread(str(p))[:, :, ::-1] for p in paths])


def _extract_smplx_body_joints_world(
    pipeline,
    tids_sorted: Iterable[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Run pipeline.smplx forward per dancer to get world-frame body joints.

    Returns ``{tid -> (frames_idx, joints_T_22_3 np.float32)}``. PromptHMR
    stores per-dancer SMPL-X parameters under
    ``results['people'][tid]['smplx_world']`` as
    ``{'pose' (T, 165), 'shape' (T, 10), 'trans' (T, 3)}``.

    The 165-dim axis-angle pose follows the standard SMPL-X layout:
    [0:3]=global_orient, [3:66]=body(21), [66:75]=jaw+leye+reye(3),
    [75:120]=left_hand(15), [120:165]=right_hand(15). The default
    ``smplx.SMPLX.forward`` reshapes every component (including
    ``self.jaw_pose`` and friends, which initialise at batch=1) and
    cats them along ``dim=1``, so we MUST pass per-frame jaw/eye
    poses or the dim-0 mismatch fires (``Expected size B but got 1``).
    Mirror the call site PromptHMR uses internally
    (``pipeline/world.py:104``): zero jaw/eye/expression, real left+
    right hand from the pose tensor.
    """
    import torch

    device = next(pipeline.smplx.parameters()).device
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for tid in tids_sorted:
        person = pipeline.results["people"][tid]
        smplx_w = person["smplx_world"]
        with torch.no_grad():
            pose = torch.as_tensor(smplx_w["pose"], dtype=torch.float32, device=device)
            shape = torch.as_tensor(smplx_w["shape"], dtype=torch.float32, device=device)
            transl = torch.as_tensor(smplx_w["trans"], dtype=torch.float32, device=device)
            B = pose.shape[0]
            zeros3 = torch.zeros(B, 3, dtype=pose.dtype, device=device)
            o = pipeline.smplx(
                global_orient=pose[:, 0:3],
                body_pose=pose[:, 3:66],
                left_hand_pose=pose[:, 75:120],
                right_hand_pose=pose[:, 120:165],
                betas=shape,
                transl=transl,
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                expression=torch.zeros(B, 10, dtype=pose.dtype, device=device),
            )
        body = o.joints[:, :22].detach().cpu().numpy().astype(np.float32)
        frames_idx = np.asarray(person["frames"], dtype=np.int64)
        out[int(tid)] = (frames_idx, body)
    return out


def _write_smpl_and_world4d(
    pipeline,
    out_dir: Path,
    tids_sorted: Sequence[int],
    n_frames: int,
) -> None:
    """Mirror Pipeline.__call__'s tail logic so we get the same .smpl/.mcs/.glb."""
    from smplcodec import SMPLCodec
    from pipeline.mcs_export_cam import export_scene_with_camera

    smpl_paths: List[Path] = []
    presence: List[List[int]] = []
    for tid in tids_sorted:
        v = pipeline.results["people"][tid]
        smpl_f = out_dir / f"subject-{tid}.smpl"
        SMPLCodec(
            shape_parameters=v["smplx_world"]["shape"].mean(0),
            body_pose=v["smplx_world"]["pose"][:, : 22 * 3].reshape(-1, 22, 3),
            body_translation=v["smplx_world"]["trans"],
            frame_count=v["frames"].shape[0],
            frame_rate=float(pipeline.cfg.fps),
        ).write(str(smpl_f))
        smpl_paths.append(smpl_f)
        presence.append([int(v["frames"][0]), int(v["frames"][-1]) + 1])

    export_scene_with_camera(
        smpl_buffers=[open(p, "rb").read() for p in smpl_paths],
        frame_presences=presence,
        num_frames=n_frames,
        output_path=str(out_dir / "world4d.mcs"),
        rotation_matrices=pipeline.results["camera_world"]["Rcw"],
        translations=pipeline.results["camera_world"]["Tcw"],
        focal_length=pipeline.results["camera_world"]["img_focal"],
        principal_point=pipeline.results["camera_world"]["img_center"],
        frame_rate=float(pipeline.cfg.fps),
        smplx_path="data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz",
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intermediates-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write results.pkl, world4d.{mcs,glb}, etc.")
    p.add_argument("--prompthmr-path", type=Path,
                   default=Path(os.environ.get("PROMPTHMR_PATH", "~/code/PromptHMR")).expanduser())
    p.add_argument("--static-camera", action="store_true",
                   help="Skip DROID-SLAM and assume identity camera (use for tripod-mounted clips)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Animation frame rate written into the .mcs/.smpl files")
    p.add_argument("--no-post-opt", action="store_true",
                   help="Skip PromptHMR's post optimisation (faster; less polished joints)")
    args = p.parse_args(argv)

    interm = args.intermediates_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    phmr_root = args.prompthmr_path.expanduser().resolve()

    ok, errs = intermediates_layout_ok(interm)
    if not ok:
        for e in errs:
            print(f"[run_promthmr_vid] ERROR: {e}", file=sys.stderr)
        return 2

    inject_prompthmr_path(phmr_root)
    chdir_to_prompthmr(phmr_root)

    from pipeline import Pipeline
    from pipeline.tools import est_camera
    from pipeline.spec import run_cam_calib

    images = _load_frames_rgb(interm / "frames")
    H, W = images[0].shape[:2]
    n_frames = len(images)

    tracks = joblib.load(interm / "tracks.pkl")
    tracks = load_per_track_masks(interm, tracks, H=H, W=W)
    union = np.load(interm / "masks_union.npy")

    pipeline = Pipeline(static_cam=args.static_camera)
    pipeline.images = images
    pipeline.seq_folder = str(out_dir)
    pipeline.cfg.seq_folder = pipeline.seq_folder
    # mask_prompt = (cfg.tracker == 'sam2') in pipeline.hps_estimation; we rely on
    # that gate to route our SAM-2 masks into PromptHMR-Vid's mask-prompt path.
    pipeline.cfg.tracker = "sam2"
    pipeline.cfg.fps = args.fps
    pipeline.fps = args.fps

    intr = _read_optional_intrinsics(interm)
    if intr is not None:
        pipeline.results = {"camera": {
            "img_focal": float(intr["fx"]),
            "img_center": np.array([intr["cx"], intr["cy"]], dtype=np.float32),
        }}
    else:
        pipeline.results = {"camera": est_camera(images[0])}

    pipeline.results.update({
        "people": tracks,
        "timings": {},
        "masks": union,
        "has_tracks": True,
        "has_hps_cam": False,
        "has_hps_world": False,
        "has_slam": False,
        "has_hands": False,
        "has_2d_kpts": False,
        "has_post_opt": False,
    })

    if pipeline.cfg.use_spec_calib:
        stride = max(1, n_frames // 30)
        spec = run_cam_calib(
            pipeline.images,
            out_folder=str(out_dir / "spec_calib"),
            save_res=True,
            stride=stride,
            method="spec",
            first_frame_idx=0,
        )
        pipeline.results["spec_calib"] = spec

    pipeline.camera_motion_estimation(args.static_camera)
    pipeline.estimate_2d_keypoints()
    pipeline.hps_estimation()
    pipeline.world_hps_estimation()

    import torch

    def _to_numpy(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _to_numpy(v)
            elif isinstance(v, torch.Tensor):
                d[k] = v.detach().cpu().numpy()
    _to_numpy(pipeline.results)

    if pipeline.cfg.run_post_opt and not args.no_post_opt:
        pipeline.post_optimization()

    joblib.dump(pipeline.results, out_dir / "results.pkl")

    tids_sorted = sorted_tid_list(pipeline.results["people"])
    pipeline.smplx = pipeline.smplx.to("cuda")
    per_track = _extract_smplx_body_joints_world(pipeline, tids_sorted)
    joints = joints_world_padded(per_track, n_frames=n_frames, tid_order=tids_sorted)
    np.save(out_dir / "joints_world.npy", joints)

    _write_smpl_and_world4d(pipeline, out_dir, tids_sorted, n_frames)

    print(
        f"[run_promthmr_vid] wrote {out_dir / 'results.pkl'}, "
        f"joints_world.npy {joints.shape}, world4d.mcs/glb, "
        f"{len(tids_sorted)} subject-*.smpl files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
