import cv2
import numpy as np
from pathlib import Path
from threed.stage_a.extract_frames import extract_frames


def _make_video(path: Path, n_frames: int = 5, w: int = 1920, h: int = 1080):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def test_extract_frames_resizes_to_max_height(tmp_path):
    src = tmp_path / "video.mp4"
    _make_video(src, n_frames=5, w=1920, h=1080)
    out_896 = tmp_path / "frames_896"
    out_full = tmp_path / "frames_full"

    n = extract_frames(src, out_896, out_full, max_height=896)
    assert n == 5

    img_896 = cv2.imread(str(out_896 / "00000000.jpg"))
    img_full = cv2.imread(str(out_full / "00000000.jpg"))
    assert img_896.shape[0] == 896
    assert img_full.shape[0] == 1080  # untouched


def test_extract_frames_caps_at_max_frames(tmp_path):
    """`max_frames=N` stops decoding after N frames so downstream Stage A
    cost stays bounded for long clips. None or 0 means no cap."""
    src = tmp_path / "video.mp4"
    _make_video(src, n_frames=20, w=640, h=360)
    out_resized = tmp_path / "frames_resized"
    out_full = tmp_path / "frames_full"

    n = extract_frames(src, out_resized, out_full, max_height=896, max_frames=8)
    assert n == 8
    assert (out_resized / "00000007.jpg").exists()
    assert not (out_resized / "00000008.jpg").exists()
    assert (out_full / "00000007.jpg").exists()
    assert not (out_full / "00000008.jpg").exists()


def test_extract_frames_max_frames_none_is_no_cap(tmp_path):
    src = tmp_path / "video.mp4"
    _make_video(src, n_frames=12, w=640, h=360)
    out_resized = tmp_path / "frames_resized"
    out_full = tmp_path / "frames_full"

    n = extract_frames(src, out_resized, out_full, max_height=896, max_frames=None)
    assert n == 12


def test_extract_frames_max_frames_larger_than_clip_is_no_op(tmp_path):
    src = tmp_path / "video.mp4"
    _make_video(src, n_frames=4, w=640, h=360)
    out_resized = tmp_path / "frames_resized"
    out_full = tmp_path / "frames_full"

    n = extract_frames(src, out_resized, out_full, max_height=896, max_frames=999)
    assert n == 4
