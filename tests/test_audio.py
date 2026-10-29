import unittest
import numpy as np
from voice_passability.audio import windows

class WindowTests(unittest.TestCase):
    def test_tail_is_filtered_before_padding(self):
        for seconds,count in [(2.99,0),(3,1),(5,1),(7,1),(8,2),(12,2),(13,3)]:
            with self.subTest(seconds=seconds):
                chunks=windows(np.ones(round(seconds*16000),dtype=np.float32))
                self.assertEqual(len(chunks),count)
                self.assertTrue(all(len(c)==80000 for c in chunks))
    def test_short_tail_cannot_change_included_windows(self):
        audio=np.ones(7*16000,dtype=np.float32)
        before=windows(audio);audio[80000:]=-20
        np.testing.assert_array_equal(before[0],windows(audio)[0])
    def test_silent_recording_keeps_one_eligible_fallback(self):
        self.assertEqual(len(windows(np.zeros(12*16000,dtype=np.float32))),1)

if __name__=='__main__':unittest.main()
