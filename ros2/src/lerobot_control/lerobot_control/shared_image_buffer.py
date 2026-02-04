"""
Shared Memory Image Buffer for Multi-Process Inference

Provides zero-copy shared memory communication between image worker processes
and the main inference process. Each camera has its own buffer slot with:
- Image data (numpy array)
- Timestamp
- Ready flag (indicates new data available)
"""

import numpy as np
from multiprocessing import shared_memory, Lock
from typing import Dict, List, Tuple, Optional
import struct
import time


class SharedImageBuffer:
    """
    Zero-copy shared memory buffer for camera images.

    Memory layout per camera:
    - Image data: H x W x 3 uint8 (e.g., 480 x 640 x 3 = 921,600 bytes)
    - Timestamp: float64 (8 bytes)
    - Frame counter: uint64 (8 bytes) - incremented on each write
    - Total per camera: image_size + 16 bytes

    Usage:
        # In main process (creates shared memory)
        buffer = SharedImageBuffer(
            camera_names=['waist', 'wrist_r', 'chest', 'wrist_l'],
            image_shape=(480, 640, 3),
            create=True
        )

        # In worker process (attaches to existing shared memory)
        buffer = SharedImageBuffer(
            camera_names=['waist', 'wrist_r', 'chest', 'wrist_l'],
            image_shape=(480, 640, 3),
            create=False
        )
    """

    # Constants for memory layout
    TIMESTAMP_SIZE = 8  # float64
    COUNTER_SIZE = 8    # uint64
    METADATA_SIZE = TIMESTAMP_SIZE + COUNTER_SIZE

    def __init__(
        self,
        camera_names: List[str],
        image_shape: Tuple[int, int, int],
        create: bool = True,
        buffer_name_prefix: str = "lerobot_img_"
    ):
        """
        Initialize shared image buffer.

        Args:
            camera_names: List of camera names (e.g., ['waist', 'wrist_r', ...])
            image_shape: Shape of images (H, W, C), e.g., (480, 640, 3)
            create: If True, create new shared memory. If False, attach to existing.
            buffer_name_prefix: Prefix for shared memory names
        """
        self.camera_names = camera_names
        self.image_shape = image_shape
        self.image_size = int(np.prod(image_shape))
        self.buffer_size = self.image_size + self.METADATA_SIZE
        self.buffer_name_prefix = buffer_name_prefix
        self.create = create

        # Shared memory blocks (one per camera)
        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}

        # Last read frame counters (to detect new frames)
        self._last_read_counters: Dict[str, int] = {name: 0 for name in camera_names}

        # Initialize shared memory
        self._init_shared_memory()

    def _get_shm_name(self, camera_name: str) -> str:
        """Get shared memory name for a camera."""
        return f"{self.buffer_name_prefix}{camera_name}"

    def _init_shared_memory(self):
        """Initialize shared memory blocks for all cameras."""
        for camera_name in self.camera_names:
            shm_name = self._get_shm_name(camera_name)

            if self.create:
                # Clean up any existing shared memory with same name
                try:
                    existing = shared_memory.SharedMemory(name=shm_name)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass

                # Create new shared memory
                shm = shared_memory.SharedMemory(
                    name=shm_name,
                    create=True,
                    size=self.buffer_size
                )
                # Initialize to zeros
                np.ndarray((self.buffer_size,), dtype=np.uint8, buffer=shm.buf).fill(0)
            else:
                # Attach to existing shared memory (with retry for startup timing)
                max_retries = 50
                for i in range(max_retries):
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        break
                    except FileNotFoundError:
                        if i < max_retries - 1:
                            time.sleep(0.1)
                        else:
                            raise RuntimeError(
                                f"Shared memory '{shm_name}' not found. "
                                "Make sure the main process creates it first."
                            )

            self._shm_blocks[camera_name] = shm

    def write(self, camera_name: str, image: np.ndarray, timestamp: float):
        """
        Write image to shared memory buffer.

        Called by image worker process after decompressing JPEG.

        Args:
            camera_name: Name of the camera
            image: Decompressed image as numpy array (H, W, 3) uint8
            timestamp: ROS2 message timestamp
        """
        if camera_name not in self._shm_blocks:
            raise ValueError(f"Unknown camera: {camera_name}")

        shm = self._shm_blocks[camera_name]

        # Validate image shape
        if image.shape != self.image_shape:
            raise ValueError(
                f"Image shape {image.shape} doesn't match expected {self.image_shape}"
            )

        # Create view of shared memory
        buf = shm.buf

        # Write image data (ensure contiguous)
        image_flat = np.ascontiguousarray(image).flatten()
        buf[:self.image_size] = image_flat.tobytes()

        # Write timestamp
        timestamp_bytes = struct.pack('d', timestamp)
        buf[self.image_size:self.image_size + self.TIMESTAMP_SIZE] = timestamp_bytes

        # Increment and write frame counter (atomic-ish via single write)
        counter_offset = self.image_size + self.TIMESTAMP_SIZE
        current_counter = struct.unpack('Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE]))[0]
        new_counter = current_counter + 1
        buf[counter_offset:counter_offset + self.COUNTER_SIZE] = struct.pack('Q', new_counter)

    def read(self, camera_name: str) -> Tuple[np.ndarray, float, int]:
        """
        Read image from shared memory buffer.

        Args:
            camera_name: Name of the camera

        Returns:
            Tuple of (image, timestamp, frame_counter)
        """
        if camera_name not in self._shm_blocks:
            raise ValueError(f"Unknown camera: {camera_name}")

        shm = self._shm_blocks[camera_name]
        buf = shm.buf

        # Read frame counter first
        counter_offset = self.image_size + self.TIMESTAMP_SIZE
        frame_counter = struct.unpack('Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE]))[0]

        # Read image data
        image_data = bytes(buf[:self.image_size])
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(self.image_shape).copy()

        # Read timestamp
        timestamp = struct.unpack('d', bytes(buf[self.image_size:self.image_size + self.TIMESTAMP_SIZE]))[0]

        return image, timestamp, frame_counter

    def has_new_frame(self, camera_name: str) -> bool:
        """Check if camera has a new frame since last read."""
        if camera_name not in self._shm_blocks:
            return False

        shm = self._shm_blocks[camera_name]
        buf = shm.buf

        counter_offset = self.image_size + self.TIMESTAMP_SIZE
        current_counter = struct.unpack('Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE]))[0]

        return current_counter > self._last_read_counters[camera_name]

    def read_if_new(self, camera_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Read image only if there's a new frame.

        Returns:
            Tuple of (image, timestamp) if new frame available, None otherwise
        """
        if not self.has_new_frame(camera_name):
            return None

        image, timestamp, frame_counter = self.read(camera_name)
        self._last_read_counters[camera_name] = frame_counter
        return image, timestamp

    def read_all_if_ready(self) -> Optional[Dict[str, Tuple[np.ndarray, float]]]:
        """
        Read all cameras only if ALL have new frames.

        This ensures synchronized observations across all cameras.

        Returns:
            Dict mapping camera_name -> (image, timestamp) if all ready, None otherwise
        """
        # Check if all cameras have new frames
        for camera_name in self.camera_names:
            if not self.has_new_frame(camera_name):
                return None

        # Read all cameras
        result = {}
        for camera_name in self.camera_names:
            image, timestamp, frame_counter = self.read(camera_name)
            self._last_read_counters[camera_name] = frame_counter
            result[camera_name] = (image, timestamp)

        return result

    def get_frame_counters(self) -> Dict[str, int]:
        """Get current frame counters for all cameras."""
        counters = {}
        for camera_name in self.camera_names:
            shm = self._shm_blocks[camera_name]
            buf = shm.buf
            counter_offset = self.image_size + self.TIMESTAMP_SIZE
            counters[camera_name] = struct.unpack(
                'Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE])
            )[0]
        return counters

    def close(self):
        """Close shared memory connections."""
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self):
        """Unlink (delete) shared memory blocks. Only call from creating process."""
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_blocks.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class SharedJointStateBuffer:
    """
    Shared memory buffer for joint states.

    Memory layout:
    - Joint positions: num_joints x float64
    - Timestamp: float64
    - Frame counter: uint64
    """

    TIMESTAMP_SIZE = 8
    COUNTER_SIZE = 8

    def __init__(
        self,
        num_joints: int,
        create: bool = True,
        buffer_name: str = "lerobot_joint_state"
    ):
        self.num_joints = num_joints
        self.positions_size = num_joints * 8  # float64
        self.buffer_size = self.positions_size + self.TIMESTAMP_SIZE + self.COUNTER_SIZE
        self.buffer_name = buffer_name
        self.create = create
        self._last_read_counter = 0

        if create:
            try:
                existing = shared_memory.SharedMemory(name=buffer_name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=buffer_name,
                create=True,
                size=self.buffer_size
            )
            np.ndarray((self.buffer_size,), dtype=np.uint8, buffer=self._shm.buf).fill(0)
        else:
            max_retries = 50
            for i in range(max_retries):
                try:
                    self._shm = shared_memory.SharedMemory(name=buffer_name)
                    break
                except FileNotFoundError:
                    if i < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        raise

    def write(self, positions: np.ndarray, timestamp: float):
        """Write joint positions to shared memory."""
        buf = self._shm.buf

        # Write positions
        positions_bytes = positions.astype(np.float64).tobytes()
        buf[:self.positions_size] = positions_bytes

        # Write timestamp
        buf[self.positions_size:self.positions_size + self.TIMESTAMP_SIZE] = struct.pack('d', timestamp)

        # Increment counter
        counter_offset = self.positions_size + self.TIMESTAMP_SIZE
        current = struct.unpack('Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE]))[0]
        buf[counter_offset:counter_offset + self.COUNTER_SIZE] = struct.pack('Q', current + 1)

    def read(self) -> Tuple[np.ndarray, float, int]:
        """Read joint positions from shared memory."""
        buf = self._shm.buf

        counter_offset = self.positions_size + self.TIMESTAMP_SIZE
        counter = struct.unpack('Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE]))[0]

        positions = np.frombuffer(bytes(buf[:self.positions_size]), dtype=np.float64).copy()
        timestamp = struct.unpack('d', bytes(buf[self.positions_size:self.positions_size + self.TIMESTAMP_SIZE]))[0]

        return positions, timestamp, counter

    def read_if_new(self) -> Optional[Tuple[np.ndarray, float]]:
        """Read only if new data available."""
        buf = self._shm.buf
        counter_offset = self.positions_size + self.TIMESTAMP_SIZE
        counter = struct.unpack('Q', bytes(buf[counter_offset:counter_offset + self.COUNTER_SIZE]))[0]

        if counter <= self._last_read_counter:
            return None

        positions, timestamp, counter = self.read()
        self._last_read_counter = counter
        return positions, timestamp

    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
