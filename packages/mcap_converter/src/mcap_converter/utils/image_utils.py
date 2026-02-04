"""Image processing utilities"""

import io
import numpy as np
import cv2
from PIL import Image


def decode_image(img_record: bytes, encoding: str, height: int, width: int) -> np.ndarray:
    """
    Decode ROS Image message data

    Args:
        img_record: Raw image bytes from ROS message
        encoding: Image encoding (e.g., 'bgr8', 'rgb8', 'mono8')
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Decoded image as numpy array (H, W, C) for RGB or (H, W) for mono

    Raises:
        ValueError: If encoding is not supported
    """
    img_data = np.frombuffer(img_record, dtype=np.uint8)

    if encoding == 'bgr8':
        img_bgr = img_data.reshape((height, width, 3))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    elif encoding == 'rgb8':
        return img_data.reshape((height, width, 3))

    elif encoding == 'mono8' or encoding == 'mono16':
        return img_data.reshape((height, width))

    elif encoding in ['jpeg', 'jpg']:
        # JPEG compressed
        return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    elif encoding in ['png']:
        # PNG compressed
        img = Image.open(io.BytesIO(img_record))
        return np.array(img)

    elif encoding in ['yuv422_yuy2', 'yuyv']:
        # YUV 4:2:2 (YUYV) format - 2 bytes per pixel
        img_yuyv = img_data.reshape((height, width, 2))
        img_rgb = cv2.cvtColor(img_yuyv, cv2.COLOR_YUV2RGB_YUYV)
        return img_rgb

    elif encoding in ['uyvy']:
        # YUV 4:2:2 (UYVY) format - 2 bytes per pixel
        img_uyvy = img_data.reshape((height, width, 2))
        img_rgb = cv2.cvtColor(img_uyvy, cv2.COLOR_YUV2RGB_UYVY)
        return img_rgb

    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")


def decode_compressed_image(data: bytes, format: str) -> np.ndarray:
    """
    Decode CompressedImage message data

    Args:
        data: Raw compressed image bytes
        format: Compression format (e.g., 'jpeg', 'png', or ROS format like 'rgb8; jpeg compressed bgr8')

    Returns:
        Decoded image as numpy array (H, W, C) in RGB format

    Raises:
        ValueError: If format is not supported
    """
    # Parse ROS CompressedImage format string
    # Format can be: 'jpeg', 'png', or 'rgb8; jpeg compressed bgr8'
    format_lower = format.lower()

    # Extract compression type from ROS format string
    if 'jpeg' in format_lower or 'jpg' in format_lower:
        compression = 'jpeg'
    elif 'png' in format_lower:
        compression = 'png'
    else:
        compression = format_lower

    img_data = np.frombuffer(data, dtype=np.uint8)

    if compression in ['jpeg', 'jpg']:
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode JPEG image")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    elif compression == 'png':
        img = Image.open(io.BytesIO(data))
        return np.array(img)

    else:
        raise ValueError(f"Unsupported compressed image format: {format}")


def encode_image_to_bytes(img: np.ndarray, format: str = 'png') -> bytes:
    """
    Encode numpy image array to bytes

    Args:
        img: Image array (H, W, C) or (H, W)
        format: Output format ('png', 'jpg', 'jpeg')

    Returns:
        Encoded image bytes
    """
    if format.lower() in ['jpg', 'jpeg']:
        success, buffer = cv2.imencode('.jpg', img)
    elif format.lower() == 'png':
        success, buffer = cv2.imencode('.png', img)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise RuntimeError("Failed to encode image")

    return buffer.tobytes()


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image to target size

    Args:
        img: Image array
        target_size: (width, height) tuple

    Returns:
        Resized image
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
