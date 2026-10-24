"""Synthetic timestamp/buffer/lifecycle controls; no weights or dataset reads."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from adaptive_pipeline.adaptive_framing.frame_manager import FramePacket, LatestFrameBuffer, CaptureError
from adaptive_pipeline.adaptive_framing.capture import VideoCaptureThread, TimestampReplayCapture
from adaptive_pipeline.run_stream import consume, check_dense_config


def packet(fid, source='s'):
    now=time.perf_counter_ns()
    return FramePacket(np.zeros((2,3,3),np.uint8),source,fid,fid*1000,now,now,now)


def rows():
    return [dict(image=str(i),frame_id=i,source_id='s',source_timestamp_ns=1000000+i*33333333)
            for i in range(4)]


class BufferTests(unittest.TestCase):
    def test_latest_atomic_replacement_and_original_ids(self):
        b=LatestFrameBuffer()
        for i in [0,3,9]:b.put_packet(packet(i))
        b.close();s=b.take()
        self.assertEqual(s.packet.frame_id,9)
        self.assertEqual(s.newest_published_id,9)
        self.assertEqual(s.pending_after,0)
        self.assertIsNone(b.take())
        self.assertEqual((b.stats()['captured'],b.stats()['consumed'],b.stats()['replaced']),(3,1,2))
        self.assertEqual(b.stats()['peak_pending'],1)
        self.assertEqual(b.stats()['pending_bytes'],0)

    def test_fifo_drains_all_after_eof(self):
        b=LatestFrameBuffer(fifo=True)
        for i in range(4):b.put_packet(packet(i))
        b.close()
        self.assertEqual([b.take().packet.frame_id for _ in range(4)],list(range(4)))
        self.assertIsNone(b.take())
        self.assertEqual(b.stats()['peak_pending'],4)
        self.assertEqual(b.stats()['replaced'],0)

    def test_eof_last_publication_cannot_be_lost(self):
        b=LatestFrameBuffer();answer=[];waiting=threading.Event()
        def reader():
            waiting.set();answer.append(b.take());answer.append(b.take())
        t=threading.Thread(target=reader);t.start();waiting.wait()
        b.put_packet(packet(4));b.close();t.join(1)
        self.assertFalse(t.is_alive())
        self.assertEqual(answer[0].packet.frame_id,4)
        self.assertIsNone(answer[1])

    def test_error_drains_then_raises_and_wakes_waiter(self):
        b=LatestFrameBuffer();b.put_packet(packet(0));b.close(ValueError('decode'))
        self.assertEqual(b.take().packet.frame_id,0)
        with self.assertRaises(CaptureError) as e:b.take()
        self.assertIsInstance(e.exception.__cause__,ValueError)
        self.assertTrue(b.stats()['failed'])

    def test_capacity_is_error_not_backpressure(self):
        b=LatestFrameBuffer(fifo=True,max_pending=1);b.put_packet(packet(0))
        with self.assertRaises(CaptureError):b.put_packet(packet(1))
        self.assertEqual(b.stats()['captured'],1)
        self.assertEqual(b.stats()['replaced'],0)

    def test_nonmonotonic_or_changed_source_rejected(self):
        for bad in [packet(0),packet(1,'other')]:
            b=LatestFrameBuffer();b.put_packet(packet(0))
            with self.assertRaises(ValueError):b.put_packet(bad)
        p=packet(2)
        b=LatestFrameBuffer();b.put_packet(p)
        from dataclasses import replace
        with self.assertRaises(ValueError):b.put_packet(replace(p,frame_id=3))

    def test_publication_after_close_fails(self):
        b=LatestFrameBuffer();b.close()
        with self.assertRaises(CaptureError):b.put_packet(packet(0))

    def test_legacy_tuple_and_image_ownership(self):
        b=LatestFrameBuffer();frame=np.zeros((2,3,3),np.uint8)
        self.assertEqual(b.put(frame),1)
        image,fid,ts=b.get()
        self.assertIs(image,frame);self.assertEqual(fid,1);self.assertIsInstance(ts,float)
        self.assertIsNone(b.get())


class CaptureTests(unittest.TestCase):
    def test_pts_scaled_without_losing_integer_source_timestamps(self):
        c=TimestampReplayCapture(rows(),LatestFrameBuffer(),90)
        self.assertEqual(c.rows[0]['source_timestamp_ns'],1000000)
        self.assertEqual(c.offsets_ns,[0,11111111,22222222,33333333])

    def test_bad_source_clock_fails_before_start(self):
        for fps in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):TimestampReplayCapture(rows(),LatestFrameBuffer(),fps)
        r=rows();r[1]['source_timestamp_ns']=r[0]['source_timestamp_ns']
        with self.assertRaises(ValueError):TimestampReplayCapture(r,LatestFrameBuffer(),15)

    def test_capture_failure_releases_and_closes(self):
        c=VideoCaptureThread('fake',LatestFrameBuffer());c._cap=Mock()
        c._loop=Mock(side_effect=IOError('bad decode'));c._launch();c.join(1)
        self.assertTrue(c.is_stopped());self.assertTrue(c.buffer.stats()['failed'])
        c._cap.release.assert_called_once()
        with self.assertRaises(CaptureError):c.buffer.take()

    def test_replay_cancel_interrupts_wait(self):
        r=rows();b=LatestFrameBuffer()
        c=TimestampReplayCapture(r,b,.01,read=lambda _:np.zeros((2,3,3),np.uint8))
        c.start();c.stop();c.join(1)
        self.assertTrue(c.is_stopped());self.assertTrue(b.stats()['closed'])
        self.assertIsNone(c.error)
        with self.assertRaises(RuntimeError):c.start()

    def test_replay_decode_failure_is_observable(self):
        c=TimestampReplayCapture(rows(),LatestFrameBuffer(),90,read=lambda _:None)
        c.start();c.join(1)
        with self.assertRaises(CaptureError):c.buffer.take()
        self.assertIsInstance(c.error,IOError)

    def test_release_even_if_decode_raises(self):
        c=VideoCaptureThread('fake',LatestFrameBuffer());c._cap=Mock()
        c._cap.read.side_effect=RuntimeError('backend')
        c.realtime=False;c.native_fps=30;c._launch();c.join(1)
        c._cap.release.assert_called_once()
        self.assertIsInstance(c.error,RuntimeError)

    def test_release_error_still_closes_and_wakes(self):
        c=VideoCaptureThread('fake',LatestFrameBuffer());c._cap=Mock()
        c._loop=Mock();c._cap.release.side_effect=RuntimeError('release')
        c._launch();c.join(1)
        self.assertTrue(c.is_stopped());self.assertTrue(c.buffer.stats()['closed'])
        with self.assertRaises(CaptureError):c.buffer.take()


class ConsumerTests(unittest.TestCase):
    def test_producer_can_replace_while_model_busy_and_drain_eof(self):
        entered=threading.Event();released=threading.Event();b=LatestFrameBuffer()
        class Capture(VideoCaptureThread):
            def start(self):return self._launch()
            def _loop(self):
                self.epoch_ns=time.perf_counter_ns();b.put_packet(packet(0))
                if not entered.wait(1):raise RuntimeError('No inference')
                for i in range(1,4):b.put_packet(packet(i))
                released.set()
        class Pipe:
            last_prediction_details=np.zeros((0,4))
            def step(self,frame,fid):
                if fid==0:
                    entered.set()
                    if not released.wait(1):raise RuntimeError('Producer was blocked')
                return np.zeros((0,6),np.float32),dict(decision='FULL')
        cap=Capture('fake',b)
        p,d,rec,s=consume(cap,Pipe(),deadline_ns=time.perf_counter_ns()+int(2e9))
        self.assertEqual([r['frame_id'] for r in rec],[0,3])
        self.assertEqual(s['buffer']['replaced'],2)
        self.assertTrue(cap.is_stopped())
        self.assertTrue(all(r['capture_ns']<=r['inference_start_ns']<=r['result_ready_ns'] for r in rec))

    def test_memory_limit_stops_capture(self):
        c=VideoCaptureThread('fake',LatestFrameBuffer());c.start=Mock()
        with patch('adaptive_pipeline.run_stream.peak_mib',return_value=4097):
            with self.assertRaises(MemoryError):consume(c,Mock(),deadline_ns=time.perf_counter_ns()+int(1e9))
        self.assertTrue(c._stop_requested.is_set())

    def test_dense_settings_guard(self):
        cfg=dict(device='cpu',half=False,imgsz=640,det_conf=.001,det_iou=.7,max_det=300,
            policy='always',use_clahe=False,keyframe_refine=False,
            class_map={'0':0,'1':1,'2':2,'3':3,'5':4,'7':5,'9':6,'11':7})
        check_dense_config(cfg)
        for key,value in [('policy','fixed'),('imgsz',960),('max_det',301)]:
            bad=dict(cfg);bad[key]=value
            with self.assertRaises(ValueError):check_dense_config(bad)


if __name__=='__main__':unittest.main()
