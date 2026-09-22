"""Continuous video and frame ingestion module.

Responsible for continuously streaming or decoding video frames from sources
(camera stream, local video file) and passing them into the frame manager.

Runs as a background daemon thread so that frame capture proceeds independently
of the inference consumer, which may be slower than the video frame rate.
"""

import threading
import time

import cv2


class VideoCaptureThread:
    """Producer thread that reads video frames and pushes them into a buffer.

    Args:
        video_path: Path to the video file (or camera index).
        buffer:     A LatestFrameBuffer instance to write frames into.
        realtime:   If True, pace frame delivery at the video's native FPS
                    to simulate a real-time source (camera / RTSP stream).
                    If False, decode frames as fast as possible (default).
    """

    def __init__(self, video_path, buffer, realtime=False):
        self.video_path = video_path
        self.buffer = buffer
        self.realtime = realtime
        self.stopped = False

        # Video metadata (populated after start)
        self.total_frames = 0
        self.native_fps = 0.0
        self.width = 0
        self.height = 0

    def start(self):
        """Launch the background capture thread."""
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
        return self

    def _capture_loop(self):
        """Continuously read frames and push into the buffer until EOF."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open {self.video_path}")
            self.stopped = True
            return

        # Store video metadata
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frame interval for real-time pacing
        frame_interval = (1.0 / self.native_fps) if (self.realtime and self.native_fps > 0) else 0

        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break
            self.buffer.put(frame)

            # If realtime pacing is enabled, sleep for the remainder of the
            # frame interval to simulate a source delivering at native FPS.
            if frame_interval > 0:
                elapsed = time.perf_counter() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()
        self.stopped = True

    def is_stopped(self):
        """Check whether the capture thread has finished reading all frames."""
        return self.stopped
