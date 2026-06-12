import os
import tempfile
import time
import unittest
from pathlib import Path

from wisperauto.postprocess_model_download import (
    XetProgress,
    _format_download_status,
    directory_size_mb,
    latest_xet_progress,
)


class PostprocessModelDownloadTests(unittest.TestCase):
    def test_directory_size_counts_incomplete_download_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / ".cache" / "huggingface" / "download"
            cache_dir.mkdir(parents=True)
            (cache_dir / "model.gguf.incomplete").write_bytes(b"x" * (2 * 1024 * 1024))
            (cache_dir / "model.gguf.lock").write_bytes(b"lock")

            self.assertGreaterEqual(directory_size_mb(cache_dir), 2.0)

    def test_latest_xet_progress_parses_recent_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            log_path = logs_dir / "xet_20260611T234509556+0200_63705.log"
            log_path.write_text(
                '{"fields":{"message":"Concurrency control for download: '
                "Current concurrency = 4; predicted bandwidth = 204800; "
                "success_ratio = 1.000; reference_size = 61.0MB; "
                "observed bytes sent so far = 104857600; "
                'completed transmissions = 12"}}\n',
                encoding="utf-8",
            )

            progress = latest_xet_progress(logs_dir=logs_dir)

            self.assertIsNotNone(progress)
            assert progress is not None
            self.assertEqual(progress.observed_bytes, 104857600)
            self.assertEqual(progress.predicted_bandwidth, 204800)
            self.assertEqual(progress.current_concurrency, 4)
            self.assertEqual(progress.completed_transmissions, 12)
            self.assertEqual(progress.log_path, log_path)

    def test_latest_xet_progress_ignores_old_logs_when_min_mtime_is_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            log_path = logs_dir / "xet_20260610T000000000+0000_1.log"
            log_path.write_text("observed bytes sent so far = 104857600\n", encoding="utf-8")
            old_time = time.time() - 3600
            os.utime(log_path, (old_time, old_time))

            self.assertIsNone(latest_xet_progress(logs_dir=logs_dir, min_mtime=time.time() - 60))

    def test_status_explains_xet_cache_when_final_file_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            target_path = target_dir / "model.gguf"
            progress = XetProgress(
                observed_bytes=150 * 1024 * 1024,
                predicted_bandwidth=256 * 1024,
                current_concurrency=8,
                completed_transmissions=3,
            )

            status = _format_download_status(60, target_path, target_dir, progress, 0)

            self.assertIn("cache Xet 150.0 Mo recus", status)
            self.assertIn("debit Xet estime 256 Ko/s", status)
            self.assertIn("fichier final peut apparaitre", status)


if __name__ == "__main__":
    unittest.main()
