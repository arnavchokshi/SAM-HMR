import numpy as np
import pickle
from pathlib import Path
from prune_tracks import FrameDetections
from threed.stage_a.extract_tracks import extract_tracks_from_cache


def _make_cache(path: Path):
    """Two tracks, ID 1 in frames 0-2, ID 2 in frames 1-3."""
    fds = []
    for f in range(4):
        xs, cs, ts = [], [], []
        if f <= 2:
            xs.append([10, 10, 100, 200]); cs.append(0.9); ts.append(1)
        if 1 <= f <= 3:
            xs.append([200, 50, 300, 250]); cs.append(0.85); ts.append(2)
        fds.append(FrameDetections(
            xyxys=np.asarray(xs, dtype=np.float32) if xs else np.zeros((0, 4), np.float32),
            confs=np.asarray(cs, dtype=np.float32),
            tids=np.asarray(ts, dtype=np.int64),
        ))
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "demo.pkl", "wb") as f:
        pickle.dump(fds, f)


def test_extract_tracks_from_cache(tmp_path):
    _make_cache(tmp_path)
    tracks = extract_tracks_from_cache(tmp_path, min_total_frames=1, min_conf=0.0)
    assert set(tracks) == {1, 2}
    np.testing.assert_array_equal(tracks[1].frames, [0, 1, 2])
    np.testing.assert_array_equal(tracks[2].frames, [1, 2, 3])
    assert tracks[1].bboxes.shape == (3, 4)


def test_extract_tracks_max_frames_drops_late_detections(tmp_path):
    """`max_frames=N` drops detections at frame index >= N so the
    track frame indices stay <N — required when Stage A also caps
    frame extraction (otherwise PHMR's images[frame_idx] OOBs)."""
    _make_cache(tmp_path)
    tracks = extract_tracks_from_cache(
        tmp_path, min_total_frames=1, min_conf=0.0, max_frames=2,
    )
    assert set(tracks) == {1, 2}
    np.testing.assert_array_equal(tracks[1].frames, [0, 1])
    np.testing.assert_array_equal(tracks[2].frames, [1])


def test_extract_tracks_max_frames_drops_track_with_no_remaining_detections(tmp_path):
    _make_cache(tmp_path)
    tracks = extract_tracks_from_cache(
        tmp_path, min_total_frames=1, min_conf=0.0, max_frames=1,
    )
    assert set(tracks) == {1}
    np.testing.assert_array_equal(tracks[1].frames, [0])


def test_extract_tracks_max_frames_none_is_no_op(tmp_path):
    _make_cache(tmp_path)
    tracks = extract_tracks_from_cache(
        tmp_path, min_total_frames=1, min_conf=0.0, max_frames=None,
    )
    assert set(tracks) == {1, 2}
    np.testing.assert_array_equal(tracks[1].frames, [0, 1, 2])
    np.testing.assert_array_equal(tracks[2].frames, [1, 2, 3])
