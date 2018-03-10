import sys
sys.path.append("../detection_pipeline")
from sliding_windows_factory import SlidingWindowsFactory

import unittest


class SlidingWindowsFactoryTests(unittest.TestCase):

    def test_should_produce_one_window_for_step_size_32(self):
        windows_amount = SlidingWindowsFactory._get_windows_amount(area_size=64, window_size=64, step_size=32)
        self.assertEqual(windows_amount, 1)

    def test_should_produce_one_window_for_step_size_48(self):
        windows_amount = SlidingWindowsFactory._get_windows_amount(area_size=64, window_size=64, step_size=48)
        self.assertEqual(windows_amount, 1)

    def test_should_produce_one_window_for_step_size_64(self):
        windows_amount = SlidingWindowsFactory._get_windows_amount(area_size=64, window_size=64, step_size=64)
        self.assertEqual(windows_amount, 1)

    def test_should_produce_two_windows_for_step_size_64(self):
        windows_amount = SlidingWindowsFactory._get_windows_amount(area_size=128, window_size=64, step_size=64)
        self.assertEqual(windows_amount, 2)

    def test_should_produce_two_windows_for_step_size_63(self):
        windows_amount = SlidingWindowsFactory._get_windows_amount(area_size=128, window_size=64, step_size=63)
        self.assertEqual(windows_amount, 2)

    def test_should_throw_exception_when_step_size_is_bigger_then_area_size(self):
        with self.assertRaises(Exception):
            SlidingWindowsFactory._get_windows_amount(area_size=64, step_size=65)

    if __name__ == "__main__":
        unittest.main()