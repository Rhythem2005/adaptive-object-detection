"""Independent capture with lifecycle management and timestamp-paced replay.

Replay capture_ns is a mapped source deadline, including subsequent JPEG decode.
Camera capture_ns is host receive time, not sensor time. Both reuse one capture
thread lifecycle. Frozen offline dataset processing never imports this module.
"""
import math
import threading
import time

import cv2

from .frame_manager import FramePacket


class VideoCaptureThread:
    def __init__(self, video_path, buffer, realtime=True):
        self.video_path, self.buffer, self.realtime = video_path, buffer, realtime
        self._stopped = threading.Event()
        self._stop_requested = threading.Event()
        self._thread = None
        self.error = None
        self.total_frames = 0
        self.native_fps = 0.0
        self.width = self.height = 0
        self._cap = None
        self.events = []
        self.epoch_ns = self.finished_ns = None
        self.backend = None

    def _launch(self):
        if self._thread is not None:
            raise RuntimeError('Capture instances cannot be restarted')
        self._thread = threading.Thread(target=self._run, name='frame-capture', daemon=True)
        self._thread.start()
        return self

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Capture instances cannot be restarted')
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self._cap.release()
            raise IOError(f'Cannot open {self.video_path}')
        self.backend = self._cap.getBackendName()
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not isinstance(self.video_path, int) and self.realtime and not (
                math.isfinite(self.native_fps) and self.native_fps > 0):
            self._cap.release()
            raise ValueError('Paced video requires finite positive FPS; verified PTS needs timestamp replay')
        return self._launch()

    def _run(self):
        try:
            self._loop()
        except BaseException as exc:
            self.error = exc
        finally:
            if self._cap is not None:
                try:
                    self._cap.release()
                except BaseException as exc:
                    if self.error is None:
                        self.error = exc
            self.finished_ns = time.perf_counter_ns()
            self.buffer.close(self.error)
            self._stopped.set()

    def _loop(self):
        camera = isinstance(self.video_path, int)
        self.epoch_ns = time.perf_counter_ns()
        fid = 0
        while not self._stop_requested.is_set():
            if self.realtime and not camera:
                deadline = self.epoch_ns + round(fid * 1e9 / self.native_fps)
                if self._stop_requested.wait(max(0, (deadline-time.perf_counter_ns())/1e9)):
                    break
            else:
                deadline = None
            start = time.perf_counter_ns()
            ok, frame = self._cap.read()
            end = time.perf_counter_ns()
            if not ok:
                if camera or (self.total_frames > 0 and fid < self.total_frames):
                    raise IOError(f'Capture read failed at source frame {fid}')
                break
            # Continuous camera reads avoid artificial sleeps, but cannot prove
            # that an upstream driver buffer is empty or establish sensor time.
            pts = (round(fid*1e9/self.native_fps) if not camera and
                   math.isfinite(self.native_fps) and self.native_fps > 0 else end)
            self.buffer.put_packet(FramePacket(frame, str(self.video_path), fid, pts,
                end if deadline is None else deadline, start, end,
                'host_receive_after_decode' if deadline is None else 'file_fps_deadline'))
            fid += 1

    def stop(self):
        """Request cancellation. An uninterruptible backend read can delay join."""
        self._stop_requested.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError('Capture backend did not stop within join timeout')

    def is_stopped(self):
        return self._stopped.is_set()


class TimestampReplayCapture(VideoCaptureThread):
    """Decode at scaled source timestamps; no unpaced preload or hidden queue.

    A late wakeup is logged, not repaired by altering timestamps or dropping
    source frames. Validation rejects unsustainable arrival timing.
    """
    def __init__(self, rows, buffer, target_fps, *, read=None):
        super().__init__('timestamp-replay', buffer, realtime=True)
        self.rows = list(rows)
        self.read = read or cv2.imread
        self.target_fps = target_fps
        if len(self.rows) < 2 or not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError('At least two frames and positive finite target FPS required')
        ids = [r['frame_id'] for r in self.rows]
        pts = [r['source_timestamp_ns'] for r in self.rows]
        if any(b <= a for a,b in zip(ids, ids[1:])) or any(b <= a for a,b in zip(pts, pts[1:])):
            raise ValueError('Replay IDs and timestamps must strictly increase')
        if len({r['source_id'] for r in self.rows}) != 1:
            raise ValueError('One source per replay')
        intervals = sorted(b-a for a,b in zip(pts, pts[1:]))
        n = len(intervals)
        median = intervals[n//2] if n%2 else (intervals[n//2-1]+intervals[n//2])/2
        self.offsets_ns = [round((p-pts[0]) * (1e9/target_fps) / median) for p in pts]
        self.total_frames = len(self.rows)
        self.native_fps = 1e9 / median
        self.backend = 'OpenCV imread / timestamp-paced JPEG replay'

    def start(self):
        return self._launch()

    def _loop(self):
        self.epoch_ns = time.perf_counter_ns() + 20_000_000
        for row, offset in zip(self.rows, self.offsets_ns):
            deadline = self.epoch_ns + offset
            if self._stop_requested.wait(max(0, (deadline-time.perf_counter_ns())/1e9)):
                break
            start = time.perf_counter_ns()
            frame = self.read(row['image'])
            end = time.perf_counter_ns()
            if frame is None:
                raise IOError(f"Cannot decode {row['image']}")
            expected_shape = row.get('shape', list(frame.shape))
            if list(frame.shape) != list(expected_shape):
                raise ValueError(f"Unexpected replay shape: {frame.shape}")
            packet = self.buffer.put_packet(FramePacket(frame, row['source_id'], row['frame_id'],
                row['source_timestamp_ns'], deadline, start, end))
            self.events.append(dict(frame_id=packet.frame_id, source_timestamp_ns=packet.source_timestamp_ns,
                capture_ns=deadline, decode_start_ns=start, decode_end_ns=end,
                published_ns=packet.published_ns, publish_lateness_ms=(packet.published_ns-deadline)/1e6))
