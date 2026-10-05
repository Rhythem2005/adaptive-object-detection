"""Threaded capture for the *live* demo mode only (run.py --realtime).

Research evaluation never uses this path: it processes every frame in
order (adaptive_framing/sequence.py) so results are deterministic and every
frame has an output to score. In live mode, frames replaced in the
LatestFrameBuffer are reported as *deadline drops*, which are kept strictly
separate from the scheduler's intentional SKIP decisions.

Changes: the source is opened in start() so metadata is valid immediately
(it was previously filled in asynchronously by the thread), and the stop
flag is a threading.Event.
"""
import threading
import time

import cv2


class VideoCaptureThread:
    def __init__(self, video_path, buffer, realtime=True):
        self.video_path = video_path
        self.buffer = buffer
        self.realtime = realtime
        self._stopped = threading.Event()
        self.total_frames = 0
        self.native_fps = 0.0
        self.width = self.height = 0
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open {self.video_path}")
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        interval = 1.0 / self.native_fps if self.realtime else 0.0
        t_next = time.perf_counter()
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            self.buffer.put(frame)
            if interval:
                t_next += interval
                sleep = t_next - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
        self._cap.release()
        self._stopped.set()

    def is_stopped(self):
        return self._stopped.is_set()
