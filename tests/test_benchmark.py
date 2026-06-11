import unittest

from wisperauto.benchmark import BenchmarkResult, format_benchmark_result, recommendation_text


class BenchmarkFormattingTest(unittest.TestCase):
    def test_recommendation_uses_fastest_backend(self):
        slow = BenchmarkResult(
            backend_id="faster-whisper",
            backend_label="faster-whisper",
            model_size="small",
            duration_seconds=30.0,
            load_seconds=1.0,
            transcribe_seconds=20.0,
            realtime_factor=1.5,
            segment_count=4,
            character_count=120,
        )
        fast = BenchmarkResult(
            backend_id="whisper.cpp",
            backend_label="whisper.cpp",
            model_size="small",
            duration_seconds=30.0,
            load_seconds=0.5,
            transcribe_seconds=10.0,
            realtime_factor=3.0,
            segment_count=5,
            character_count=140,
        )

        self.assertIn("whisper.cpp", recommendation_text([fast, slow]))
        self.assertIn("3.00x", recommendation_text([fast, slow]))

    def test_format_benchmark_result_is_readable(self):
        result = BenchmarkResult(
            backend_id="mlx-whisper",
            backend_label="MLX Mac",
            model_size="large-v3-turbo",
            duration_seconds=90.0,
            load_seconds=2.25,
            transcribe_seconds=45.0,
            realtime_factor=2.0,
            segment_count=12,
            character_count=900,
        )

        text = format_benchmark_result(result)

        self.assertIn("MLX Mac", text)
        self.assertIn("chargement 2.2s", text)
        self.assertIn("2.00x temps reel", text)


if __name__ == "__main__":
    unittest.main()
