"""Latest-frame buffer; FIFO is a bounded validation control.

Legacy put/get tuples remain available. Timestamped take() drains EOF under the
same lock. Image ownership transfers at put; producers must not mutate it later.
"""
from collections import deque
from dataclasses import dataclass, replace
import threading
import time


@dataclass(frozen=True)
class FramePacket:
    frame: object
    source_id: str
    frame_id: int
    source_timestamp_ns: int
    capture_ns: int
    decode_start_ns: int
    decode_end_ns: int
    timestamp_kind: str = 'replay_source_deadline'
    published_ns: int = 0


@dataclass(frozen=True)
class SelectedFrame:
    packet: FramePacket
    selected_ns: int
    newest_published_id: int
    pending_after: int


class CaptureError(RuntimeError):
    pass


class LatestFrameBuffer:
    """One pending frame, atomically replaced by the newest publication.

    A full FIFO control raises instead of slowing the producer or silently
    dropping data. take() snapshots newest_published_id at its selection lock.
    """
    def __init__(self, *, fifo=False, max_pending=300):
        if max_pending < 1:
            raise ValueError('max_pending must be positive')
        self._condition = threading.Condition()
        self._frames = deque()
        self._fifo = fifo
        self._capacity = max_pending if fifo else 1
        self._closed = False
        self._error = None
        self._source_id = None
        self._last_id = -1
        self._last_pts = self._last_capture = None
        self._legacy_id = 0
        self.captured_count = self.consumed_count = self.replaced_count = 0
        self._peak_pending = self._pending_bytes = self._peak_bytes = 0

    def put_packet(self, packet):
        with self._condition:
            if self._closed:
                raise CaptureError('Publication after buffer closure')
            if (packet.frame_id <= self._last_id or
                (self._source_id is not None and packet.source_id != self._source_id) or
                (self._last_pts is not None and packet.source_timestamp_ns <= self._last_pts) or
                (self._last_capture is not None and packet.capture_ns <= self._last_capture)):
                raise ValueError('Nonmonotonic IDs/timestamps or source change; use a fresh stream')
            if not (packet.capture_ns <= packet.decode_end_ns and
                    packet.decode_start_ns <= packet.decode_end_ns):
                raise ValueError('Invalid capture/decode timestamps')
            if self._fifo and len(self._frames) >= self._capacity:
                raise CaptureError('FIFO capacity exceeded; source must not be slowed')
            if not self._fifo and self._frames:
                old = self._frames.pop()
                self._pending_bytes -= old.frame.nbytes
                self.replaced_count += 1
            packet = replace(packet, published_ns=time.perf_counter_ns())
            self._frames.append(packet)
            self._pending_bytes += packet.frame.nbytes
            self._peak_pending = max(self._peak_pending, len(self._frames))
            self._peak_bytes = max(self._peak_bytes, self._pending_bytes)
            self._source_id, self._last_id = packet.source_id, packet.frame_id
            self._last_pts, self._last_capture = packet.source_timestamp_ns, packet.capture_ns
            self.captured_count += 1
            self._condition.notify()
            return packet

    def take(self, timeout=None):
        """Drain pending frames even after EOF, then return None or raise error."""
        with self._condition:
            available = self._condition.wait_for(lambda: self._frames or self._closed, timeout)
            if not available:
                return None
            if self._frames:
                packet = self._frames.popleft()
                self._pending_bytes -= packet.frame.nbytes
                self.consumed_count += 1
                return SelectedFrame(packet, time.perf_counter_ns(), self._last_id, len(self._frames))
            if self._error is not None:
                raise CaptureError('Capture failed') from self._error
            return None

    def close(self, error=None):
        with self._condition:
            self._closed = True
            if error is not None and self._error is None:
                self._error = error
            self._condition.notify_all()

    def put(self, frame):
        """Legacy API: host receive time after decode, generated one-based ID."""
        self._legacy_id += 1
        now = time.perf_counter_ns()
        self.put_packet(FramePacket(frame, 'legacy', self._legacy_id, now, now, now, now,
                                    'host_receive_after_decode'))
        return self._legacy_id

    def get(self):
        selection = self.take(timeout=0)
        if selection is None:
            return None
        p = selection.packet
        return p.frame, p.frame_id, p.capture_ns / 1e9

    def stats(self):
        with self._condition:
            return dict(captured=self.captured_count, consumed=self.consumed_count,
                        replaced=self.replaced_count, pending=len(self._frames),
                        peak_pending=self._peak_pending, pending_bytes=self._pending_bytes,
                        peak_pending_bytes=self._peak_bytes, closed=self._closed,
                        failed=self._error is not None, newest_published_id=self._last_id,
                        policy='fifo' if self._fifo else 'latest')
