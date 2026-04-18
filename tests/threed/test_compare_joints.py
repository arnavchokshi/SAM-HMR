"""Unit tests for :mod:`threed.compare.joints` (plan Task 11a).

These tests pin down the SMPL-22 -> COCO-17 and MHR70 -> COCO-17
joint reductions used by Stage D so that PromptHMR-Vid (SMPL-X body)
and SAM-Body4D (MHR70 body) joints can be compared on a common
17-keypoint skeleton.
"""
from __future__ import annotations

import numpy as np
import pytest

from threed.compare.joints import (
    COCO17_NAMES,
    MHR70_TO_COCO17,
    SMPL22_TO_COCO17,
    mhr70_to_coco17,
    smpl22_to_coco17,
)


class TestSmpl22ToCoco17Indices:
    """Pin down the SMPL-22 -> COCO-17 index map.

    SMPL puts LEFT body parts at *odd* indices and RIGHT parts at *even*
    indices (left_hip=1, right_hip=2, ..., left_shoulder=16,
    right_shoulder=17). The plan §11 sample mapping had L/R swapped at
    the shoulders; we fix it here and document the deviation in the
    agent log.

    SMPL has no nose / eyes / ears (head=15 is the only face joint), so
    COCO 0..4 (face) all collapse to SMPL[15]. This is a coarse
    approximation — joint-level MPJPE on the face is therefore
    artificially zero for both pipelines and should not be reported.
    """

    def test_length(self):
        assert len(SMPL22_TO_COCO17) == 17

    def test_face_collapses_to_head(self):
        for coco_face_idx in range(5):
            assert SMPL22_TO_COCO17[coco_face_idx] == 15, (
                f"COCO {COCO17_NAMES[coco_face_idx]} should map to SMPL head (15) "
                f"because SMPL has no face joints, got "
                f"{SMPL22_TO_COCO17[coco_face_idx]}"
            )

    def test_shoulders_left_first(self):
        assert SMPL22_TO_COCO17[5] == 16, "left_shoulder -> SMPL 16 (left in SMPL)"
        assert SMPL22_TO_COCO17[6] == 17, "right_shoulder -> SMPL 17 (right in SMPL)"

    def test_elbows_left_first(self):
        assert SMPL22_TO_COCO17[7] == 18, "left_elbow -> SMPL 18"
        assert SMPL22_TO_COCO17[8] == 19, "right_elbow -> SMPL 19"

    def test_wrists_left_first(self):
        assert SMPL22_TO_COCO17[9] == 20, "left_wrist -> SMPL 20"
        assert SMPL22_TO_COCO17[10] == 21, "right_wrist -> SMPL 21"

    def test_hips_left_first(self):
        assert SMPL22_TO_COCO17[11] == 1, "left_hip -> SMPL 1"
        assert SMPL22_TO_COCO17[12] == 2, "right_hip -> SMPL 2"

    def test_knees_left_first(self):
        assert SMPL22_TO_COCO17[13] == 4, "left_knee -> SMPL 4"
        assert SMPL22_TO_COCO17[14] == 5, "right_knee -> SMPL 5"

    def test_ankles_left_first(self):
        assert SMPL22_TO_COCO17[15] == 7, "left_ankle -> SMPL 7"
        assert SMPL22_TO_COCO17[16] == 8, "right_ankle -> SMPL 8"

    def test_indices_in_range(self):
        assert all(0 <= i < 22 for i in SMPL22_TO_COCO17)


class TestMhr70ToCoco17Indices:
    """Pin down the MHR70 -> COCO-17 index map.

    Composed from upstream's
    ``MHR70_TO_OPENPOSE`` (in ``models/sam_3d_body/sam_3d_body/metadata/__init__.py``)
    and ``OPENPOSE_TO_COCO`` (same file). Spot-checks against the upstream
    table:

    * COCO 0 (nose)         -> OpenPose 0 -> MHR 0
    * COCO 5 (lshoulder)    -> OpenPose 5 -> MHR 5
    * COCO 6 (rshoulder)    -> OpenPose 2 -> MHR 6
    * COCO 9 (lwrist)       -> OpenPose 7 -> MHR 62
    * COCO 10 (rwrist)      -> OpenPose 4 -> MHR 41
    * COCO 11 (lhip)        -> OpenPose 12 -> MHR 9
    * COCO 12 (rhip)        -> OpenPose 9 -> MHR 10
    """

    def test_length(self):
        assert len(MHR70_TO_COCO17) == 17

    def test_nose(self):
        assert MHR70_TO_COCO17[0] == 0

    def test_shoulders(self):
        assert MHR70_TO_COCO17[5] == 5
        assert MHR70_TO_COCO17[6] == 6

    def test_wrists(self):
        assert MHR70_TO_COCO17[9] == 62
        assert MHR70_TO_COCO17[10] == 41

    def test_hips(self):
        assert MHR70_TO_COCO17[11] == 9
        assert MHR70_TO_COCO17[12] == 10

    def test_indices_in_range(self):
        assert all(0 <= i < 70 for i in MHR70_TO_COCO17)


class TestSmpl22ToCoco17Reduction:
    def test_reduces_last_axis(self):
        joints = np.random.rand(10, 5, 22, 3).astype(np.float32)
        out = smpl22_to_coco17(joints)
        assert out.shape == (10, 5, 17, 3)

    def test_picks_correct_indices(self):
        joints = np.zeros((1, 1, 22, 3), dtype=np.float32)
        for i in range(22):
            joints[0, 0, i] = [i, i + 100, i + 200]
        out = smpl22_to_coco17(joints)
        for c, s in enumerate(SMPL22_TO_COCO17):
            np.testing.assert_array_equal(out[0, 0, c], joints[0, 0, s])

    def test_propagates_nan(self):
        joints = np.full((1, 1, 22, 3), np.nan, dtype=np.float32)
        out = smpl22_to_coco17(joints)
        assert np.isnan(out).all()

    def test_rejects_wrong_axis_size(self):
        joints = np.zeros((1, 1, 17, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="22"):
            smpl22_to_coco17(joints)


class TestMhr70ToCoco17Reduction:
    def test_reduces_last_axis(self):
        joints = np.random.rand(8, 3, 70, 3).astype(np.float32)
        out = mhr70_to_coco17(joints)
        assert out.shape == (8, 3, 17, 3)

    def test_picks_correct_indices(self):
        joints = np.zeros((1, 1, 70, 3), dtype=np.float32)
        for i in range(70):
            joints[0, 0, i] = [i, i + 100, i + 200]
        out = mhr70_to_coco17(joints)
        for c, m in enumerate(MHR70_TO_COCO17):
            np.testing.assert_array_equal(out[0, 0, c], joints[0, 0, m])

    def test_rejects_wrong_axis_size(self):
        joints = np.zeros((1, 1, 17, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="70"):
            mhr70_to_coco17(joints)
