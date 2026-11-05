import unittest
from experiments.run_single import attach_render_timing, SingleRunError
from experiments.timing import timing_lines
from experiments.run_experiments import validate_timing, RunnerError


class TimingTests(unittest.TestCase):
    def test_render_boundary_uses_shared_clock_not_process_duration(self):
        record = {"result_ready_monotonic_ms": 1000.0, "total_compute_time_ms": 30.0}
        attach_render_timing(record, "FRAME_READY_MONOTONIC_MS=1025.125000\nTotal: 999ms\n")
        self.assertEqual(record["draw_time_ms"], 25.125)
        self.assertEqual(record["total_with_draw_time_ms"], 55.125)

    def test_missing_or_inverted_boundary_is_not_fabricated(self):
        record = {"total_compute_time_ms": 30.0}
        attach_render_timing(record, "FRAME_READY_MONOTONIC_MS=10\n")
        self.assertNotIn("draw_time_ms", record)
        record["result_ready_monotonic_ms"] = 20
        with self.assertRaises(SingleRunError):
            attach_render_timing(record, "FRAME_READY_MONOTONIC_MS=10\n")

    def test_bad_iteration_measurements_are_rejected_before_legacy_validation(self):
        for times, total in [([1, -1], 0), ([1], 1), ([1, 2], 4), (None, 2)]:
            with self.subTest(times=times, total=total), self.assertRaises(RunnerError):
                validate_timing({"iterations": 2, "iteration_times_ms": times, "iteration_total_time_ms": total})

    def test_cpu_order_and_no_gpu_metrics(self):
        text = "\n".join(timing_lines({"method": "sgd", "iteration_times_ms": [1, 2],
            "iteration_total_time_ms": 3, "total_compute_time_ms": 7}))
        self.assertLess(text.index("反復時間:"), text.index("反復時間合計:"))
        self.assertLess(text.index("反復時間合計:"), text.index("総計算時間:"))
        for absent in ["GPU準備時間", "初期転送時間", "結果取得時間"]:
            self.assertNotIn(absent, text)
