"""Tests for image utility functions.

Tests the decoding/encoding contract for image formats used in the pipeline.
All image data is synthetic numpy arrays — no MCAP files required.
"""

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from mcap_converter.utils.image_utils import (
    decode_compressed_image,
    decode_image,
    encode_image_to_bytes,
    resize_image,
)

# Small fixed dimensions for fast tests
H, W = 4, 6


def make_rgb_image() -> np.ndarray:
    """Reproducible RGB test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def rgb_to_jpeg_bytes(img_rgb: np.ndarray) -> bytes:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr)
    return buf.tobytes()


def rgb_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()


class TestDecodeImage:
    def test_rgb8(self):
        img = make_rgb_image()
        result = decode_image(img.tobytes(), "rgb8", H, W)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, img)

    def test_bgr8_channels_swapped_to_rgb(self):
        # Known pixel: B=30, G=20, R=10 in BGR storage
        img_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        img_bgr[0, 0] = [30, 20, 10]  # OpenCV order: B, G, R
        result = decode_image(img_bgr.tobytes(), "bgr8", H, W)
        assert result.shape == (H, W, 3)
        assert result[0, 0, 0] == 10  # R channel
        assert result[0, 0, 1] == 20  # G channel
        assert result[0, 0, 2] == 30  # B channel

    def test_mono8(self):
        img = np.random.default_rng(1).integers(0, 256, (H, W), dtype=np.uint8)
        result = decode_image(img.tobytes(), "mono8", H, W)
        assert result.shape == (H, W)

    def test_jpeg_returns_correct_shape(self):
        img = make_rgb_image()
        jpeg_bytes = rgb_to_jpeg_bytes(img)
        result = decode_image(jpeg_bytes, "jpeg", H, W)
        assert result.shape == (H, W, 3)

    def test_png_exact_roundtrip(self):
        img = make_rgb_image()
        png_bytes = rgb_to_png_bytes(img)
        result = decode_image(png_bytes, "png", H, W)
        assert result.shape == (H, W, 3)
        np.testing.assert_array_equal(result, img)

    def test_unsupported_encoding_raises_value_error(self):
        dummy = bytes(H * W * 3)
        with pytest.raises(ValueError, match="Unsupported image encoding"):
            decode_image(dummy, "xyz_unknown_format", H, W)


class TestDecodeCompressedImage:
    def test_jpeg_bytes(self):
        img = make_rgb_image()
        result = decode_compressed_image(rgb_to_jpeg_bytes(img), "jpeg")
        assert result.shape == (H, W, 3)

    def test_png_exact_roundtrip(self):
        img = make_rgb_image()
        result = decode_compressed_image(rgb_to_png_bytes(img), "png")
        assert result.shape == (H, W, 3)
        np.testing.assert_array_equal(result, img)

    def test_ros_format_string_parsed_as_jpeg(self):
        """'rgb8; jpeg compressed bgr8' is a valid ROS CompressedImage format string."""
        img = make_rgb_image()
        result = decode_compressed_image(rgb_to_jpeg_bytes(img), "rgb8; jpeg compressed bgr8")
        assert result.shape == (H, W, 3)

    def test_unsupported_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported compressed image format"):
            decode_compressed_image(b"\x00" * 16, "bmp")


class TestResizeImage:
    def test_resize_down(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = resize_image(img, (320, 240))
        assert result.shape == (240, 320, 3)

    def test_resize_preserves_channel_count(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = resize_image(img, (50, 75))
        assert result.shape[2] == 3


class TestEncodeImageToBytes:
    def test_png_encode_returns_nonempty_bytes(self):
        img = make_rgb_image()
        result = encode_image_to_bytes(img, "png")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_jpeg_encode_returns_nonempty_bytes(self):
        img = make_rgb_image()
        result = encode_image_to_bytes(img, "jpeg")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_unsupported_format_raises_value_error(self):
        img = make_rgb_image()
        with pytest.raises(ValueError, match="Unsupported format"):
            encode_image_to_bytes(img, "bmp")
