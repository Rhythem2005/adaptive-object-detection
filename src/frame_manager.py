"""Latest-frame buffer management.

Implements a single-frame buffer (capacity = 1) ensuring the detection model
always consumes the most recent frame, dropping stale frames to minimize latency.

Design:
    - Producer thread calls put() with each newly captured frame.
    - Consumer thread calls get() to retrieve the latest unread frame.
    - If a new frame arrives before the previous one was consumed, the older
      frame is silently replaced (counted as a dropped/replaced frame).
    - Thread safety is provided by a threading.Lock guarding the shared slot.
"""

import threading
import time


class LatestFrameBuffer:
    """Thread-safe single-slot buffer that always holds the latest frame.

    Attributes:
        captured_count:  Total frames written into the buffer by the producer.
        consumed_count:  Total frames read from the buffer by the consumer.
        replaced_count:  Frames that were overwritten before being consumed.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None       # (frame_array, frame_id, capture_ts)
        self._frame_id = 0       # Monotonically increasing per-put counter

        # Counters for verification / metrics
        self.captured_count = 0
        self.consumed_count = 0
        self.replaced_count = 0

    def put(self, frame):
        """Store a newly captured frame, replacing any unconsumed frame.

        Args:
            frame: numpy array (BGR image from cv2.VideoCapture).

        Returns:
            int: The frame_id assigned to this frame.
        """
        capture_ts = time.perf_counter()
        with self._lock:
            self._frame_id += 1
            fid = self._frame_id

            # If there is an unconsumed frame in the slot, it becomes replaced
            if self._frame is not None:
                self.replaced_count += 1

            self._frame = (frame, fid, capture_ts)
            self.captured_count += 1

        return fid

    def get(self):
        """Retrieve and clear the latest frame from the buffer.

        Returns:
            tuple (frame, frame_id, capture_ts) if a new frame is available,
            or None if the buffer is empty (already consumed or not yet filled).
        """
        with self._lock:
            if self._frame is None:
                return None

            data = self._frame
            self._frame = None
            self.consumed_count += 1

        return data

    def stats(self):
        """Return a snapshot of buffer counters (thread-safe).

        Returns:
            dict with keys: captured, consumed, replaced.
        """
        with self._lock:
            return {
                "captured": self.captured_count,
                "consumed": self.consumed_count,
                "replaced": self.replaced_count,
            }
