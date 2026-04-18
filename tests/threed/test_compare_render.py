"""Unit tests for :mod:`threed.compare.render` (plan Task 11c)."""
from __future__ import annotations

import numpy as np
import pytest

from threed.compare.render import resize_keep_ratio, stitch_side_by_side


@pytest.fixture
def red_left() -> np.ndarray:
    """A 32x32 red BGR image (red is (0,0,255) in BGR)."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 2] = 255
    return img


@pytest.fixture
def green_right() -> np.ndarray:
    """A 32x32 green BGR image (green is (0,255,0) in BGR)."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 1] = 255
    return img


class TestStitchSideBySide:
    def test_output_shape_includes_gutter(self, red_left, green_right):
        out = stitch_side_by_side(red_left, green_right, gutter_px=10)
        assert out.shape == (32, 74, 3), "32 + 10 + 32 = 74 wide"

    def test_left_panel_preserved(self, red_left, green_right):
        out = stitch_side_by_side(red_left, green_right, gutter_px=10)
        np.testing.assert_array_equal(out[:, :32], red_left)

    def test_right_panel_preserved(self, red_left, green_right):
        out = stitch_side_by_side(red_left, green_right, gutter_px=10)
        np.testing.assert_array_equal(out[:, 42:], green_right)

    def test_gutter_is_black(self, red_left, green_right):
        out = stitch_side_by_side(red_left, green_right, gutter_px=10)
        assert (out[:, 32:42] == 0).all()

    def test_zero_gutter(self, red_left, green_right):
        out = stitch_side_by_side(red_left, green_right, gutter_px=0)
        assert out.shape == (32, 64, 3)
        np.testing.assert_array_equal(out[:, :32], red_left)
        np.testing.assert_array_equal(out[:, 32:], green_right)

    def test_size_mismatch_raises(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.zeros((10, 11, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="size"):
            stitch_side_by_side(a, b)


class TestResizeKeepRatio:
    def test_no_op_when_already_target(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out = resize_keep_ratio(img, target_h=100, target_w=200)
        assert out.shape == (100, 200, 3)

    def test_letterboxes_wider_target(self):
        """100x100 image into 100x200 box -> centred with black bars."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        out = resize_keep_ratio(img, target_h=100, target_w=200)
        assert out.shape == (100, 200, 3)
        assert (out[:, 50:150] == 128).all(), "image content centred"
        assert (out[:, :50] == 0).all() and (out[:, 150:] == 0).all(), "black bars"

    def test_pillarboxes_taller_target(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        out = resize_keep_ratio(img, target_h=200, target_w=100)
        assert out.shape == (200, 100, 3)
        assert (out[50:150, :] == 128).all()
        assert (out[:50, :] == 0).all() and (out[150:, :] == 0).all()

    def test_downscales_proportionally(self):
        img = np.full((400, 200, 3), 128, dtype=np.uint8)
        out = resize_keep_ratio(img, target_h=100, target_w=100)
        assert out.shape == (100, 100, 3)
        assert (out[:, 25:75] == 128).all()
