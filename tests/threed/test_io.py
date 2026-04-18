import numpy as np
from threed.io import save_tracks, load_tracks, TrackEntry


def test_tracks_roundtrip(tmp_path):
    tid = 7
    track = TrackEntry(
        track_id=tid,
        frames=np.array([0, 1, 2, 5, 6], dtype=np.int64),
        bboxes=np.random.rand(5, 4).astype(np.float32) * 1000,
        confs=np.array([0.9, 0.8, 0.85, 0.92, 0.88], dtype=np.float32),
    )
    out = tmp_path / "tracks.pkl"
    save_tracks({tid: track}, out)
    loaded = load_tracks(out)
    assert set(loaded.keys()) == {tid}
    np.testing.assert_array_equal(loaded[tid].frames, track.frames)
    np.testing.assert_allclose(loaded[tid].bboxes, track.bboxes)
