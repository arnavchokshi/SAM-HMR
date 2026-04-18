"""SMPL-22 / MHR70 -> COCO-17 joint reductions for Stage D.

Both pipelines emit body-joint trees in *different* skeleton conventions:

- **PromptHMR-Vid** outputs SMPL-X parameters which we then forward through
  ``smplx.SMPLX`` to get 22 body joints (SMPL body subset, indices 0..21
  of the joint output tensor). See
  ``threed/sidecar_promthmr/run_promthmr_vid.py:_extract_smplx_body_joints_world``.
- **SAM-Body4D** outputs MHR70 keypoints (70 joints of the Momentum
  Human Rig) — see
  ``models/sam_3d_body/sam_3d_body/metadata/__init__.py:MHR70_TO_OPENPOSE``.

To compare them, we reduce both to a 17-keypoint COCO subset (the
standard "person keypoints" layout used by COCO 2017). This module
exposes pure index lookup tables + slicing helpers so the per-pipeline
extraction code (``compare/joints_extract.py``) and the metric
computation (``compare/run_compare.py``) can stay in lock-step.

**SMPL → COCO-17 caveat (face).** SMPL's body subset has no nose, eyes
or ears — only ``head`` (joint 15). We therefore collapse COCO 0..4
(``nose``, ``left_eye``, ``right_eye``, ``left_ear``, ``right_ear``) all
onto SMPL ``head``, which means MPJPE on those five COCO indices is
artificially zero between the two pipelines (both report the head
position). Stage D reports per-joint metrics so a downstream reader can
mask the face indices out; we do NOT silently drop them so the
final metrics file is shape-stable across clips.

**Plan §11 deviation (left/right).** The plan's narrative sample
mapping had SMPL left/right swapped at the shoulders/elbows/wrists
because it implicitly assumed SMPL puts RIGHT first. SMPL actually puts
LEFT first (left_hip=1, right_hip=2, ..., left_shoulder=16,
right_shoulder=17 — see ``smplx.SMPLX`` joint names). This module
follows the *correct* SMPL convention and the deviation is recorded in
the agent log.
"""
from __future__ import annotations

from typing import List

import numpy as np


COCO17_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


SMPL22_TO_COCO17: List[int] = [
    15, 15, 15, 15, 15,
    16, 17, 18, 19, 20, 21,
    1,  2,  4,  5,  7,  8,
]


_OPENPOSE_TO_COCO_BODY25: List[int] = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
_MHR70_TO_OPENPOSE: dict = {
    0: 0, 1: 69, 2: 6, 3: 8, 4: 41,
    5: 5, 6: 7, 7: 62,
    9: 10, 10: 12, 11: 14,
    12: 9, 13: 11, 14: 13,
    15: 2, 16: 1, 17: 4, 18: 3,
    19: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20,
}
MHR70_TO_COCO17: List[int] = [_MHR70_TO_OPENPOSE[op] for op in _OPENPOSE_TO_COCO_BODY25]


def smpl22_to_coco17(joints: np.ndarray) -> np.ndarray:
    """Reduce a ``(..., 22, 3)`` SMPL body joint tensor to ``(..., 17, 3)``.

    Index along the second-to-last axis using :data:`SMPL22_TO_COCO17`.
    NaN entries propagate. Raises ``ValueError`` if the joint axis is
    not 22 (defensive — protects against accidentally feeding the
    full SMPL-X 127-joint output).
    """
    joints = np.asarray(joints)
    if joints.shape[-2] != 22:
        raise ValueError(
            f"smpl22_to_coco17 expected joint axis size 22, got {joints.shape[-2]}"
        )
    return joints[..., SMPL22_TO_COCO17, :]


def mhr70_to_coco17(joints: np.ndarray) -> np.ndarray:
    """Reduce a ``(..., 70, 3)`` MHR70 keypoint tensor to ``(..., 17, 3)``.

    Index along the second-to-last axis using :data:`MHR70_TO_COCO17`.
    NaN entries propagate. Raises ``ValueError`` if the joint axis is
    not 70 (defensive — protects against accidentally feeding
    ``pred_joint_coords`` which has a different layout).
    """
    joints = np.asarray(joints)
    if joints.shape[-2] != 70:
        raise ValueError(
            f"mhr70_to_coco17 expected joint axis size 70, got {joints.shape[-2]}"
        )
    return joints[..., MHR70_TO_COCO17, :]
