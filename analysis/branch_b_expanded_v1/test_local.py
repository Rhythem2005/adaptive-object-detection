"""Infrastructure contract checks using synthetic packets, never YOLO."""
import time
import unittest
import numpy as np
from adaptive_pipeline.adaptive_framing.capture import TimestampReplayCapture
from adaptive_pipeline.adaptive_framing.frame_manager import LatestFrameBuffer
from adaptive_pipeline.run_stream import consume
from analysis.latest_frame_fixed2_validation import held_outputs


class InfrastructureTests(unittest.TestCase):
    def test_more_than_300_packets_fifo_and_latest_preserve_metadata(self):
        rows=[dict(source_id='synthetic',frame_id=i,source_timestamp_ns=1_000_000_000+i*33_300_000,
            image=str(i),shape=[8,8,3]) for i in range(317)]
        class Processor:
            last_prediction_details=np.zeros((0,4))
            def step(self,frame,frame_id):
                time.sleep(.0003)
                return np.zeros((0,6),np.float32),dict(decision='FULL')
        for fifo in [True,False]:
            capture=TimestampReplayCapture(rows,LatestFrameBuffer(fifo=fifo,max_pending=len(rows)),10000,
                read=lambda _:np.zeros((8,8,3),np.uint8))
            p,d,records,summary=consume(capture,Processor(),deadline_ns=time.perf_counter_ns()+3_000_000_000)
            self.assertEqual(summary['buffer']['captured'],317)
            self.assertEqual(records[-1]['frame_id'],316)
            self.assertEqual(summary['buffer']['captured'],len(records)+summary['buffer']['replaced'])
            for r in records:self.assertEqual(r['source_timestamp_ns'],rows[r['frame_id']]['source_timestamp_ns'])
            if fifo:
                self.assertEqual([r['frame_id'] for r in records],list(range(317)))
                self.assertEqual(summary['buffer']['replaced'],0)
            else:
                self.assertEqual(summary['buffer']['peak_pending'],1)
                for r in records:self.assertEqual(r['frame_id'],r['newest_published_id_at_selection'])

    def test_held_output_initial_empty_and_exact_completion_boundary(self):
        a=np.ones((1,6),np.float32);b=np.full((1,6),2,np.float32)
        records=[dict(key='s/0',frame_id=0,capture_ns=100,result_ready_ns=150,decision='FULL'),
                 dict(key='s/2',frame_id=2,capture_ns=200,result_ready_ns=250,decision='SKIP')]
        events=[dict(frame_id=i,capture_ns=t) for i,t in enumerate([100,150,200,250])]
        arrays,assign=held_outputs(records,events,{'s/0':a,'s/2':b})
        self.assertEqual(arrays[0].shape,(0,6));self.assertIs(arrays[1],a);self.assertIs(arrays[2],a);self.assertIs(arrays[3],b)
        self.assertEqual([r['result_source_frame_id'] for r in assign],[None,0,0,2])


if __name__=='__main__':unittest.main()
