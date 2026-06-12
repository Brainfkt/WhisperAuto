import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from wisperauto.cancel import CancellationToken
from wisperauto.config import AppConfig
from wisperauto.errors import PostProcessUnavailableError
from wisperauto.jobs import STATUS_CANCELLED, STATUS_DONE, STATUS_ERROR, STATUS_POSTPROCESSING, STATUS_READY, JobStore
from wisperauto.pipeline import TranscriptionPipeline, safe_float
from wisperauto.postprocess import MODE_RAW, MODE_REPORT, MODE_SMART, PostProcessResult
from wisperauto.postprocess_llm import PostProcessProgress


class FakeSegment:
    def __init__(self, text: str, end: float):
        self.text = text
        self.end = end


class FakeInfo:
    duration = 2.0


class FakeEngine:
    def transcribe(self, audio_path: Path) -> tuple[list[FakeSegment], FakeInfo]:
        del audio_path
        return (
            [
                FakeSegment("Bonjour virgule maitre point", 1.0),
                FakeSegment("sauter ligne", 2.0),
            ],
            FakeInfo(),
        )


def fake_engine_factory(config: AppConfig, allow_model_download: bool) -> FakeEngine:
    del config, allow_model_download
    return FakeEngine()


def no_wait_until_file_ready(
    path: Path,
    stable_checks: int = 2,
    interval_seconds: float = 0.5,
    timeout_seconds: int = 120,
    cancel_token: CancellationToken | None = None,
) -> None:
    del path, stable_checks, interval_seconds, timeout_seconds, cancel_token


class FakePostProcessor:
    def __init__(self):
        self.calls = 0

    def build_outputs(
        self,
        raw_text: str,
        *,
        progress_callback: Callable[[PostProcessProgress], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> PostProcessResult:
        del cancel_token
        self.calls += 1
        if progress_callback:
            progress_callback(
                PostProcessProgress(
                    stage="chunk_done",
                    detail="1/1 segment(s) traite(s)",
                    total_chunks=1,
                    completed_chunks=1,
                    elapsed_seconds=1,
                    eta_seconds=0,
                )
            )
        return PostProcessResult(
            outputs={
                MODE_RAW: raw_text.strip() + "\n",
                MODE_SMART: "Bonjour, maitre.\n\n",
                MODE_REPORT: "Compte rendu structure\n\nBonjour, maitre.\n",
            },
            actions=["Post-traitement LLM local : 1 segment(s) traite(s)."],
        )


class FailingPostProcessor:
    def build_outputs(
        self,
        raw_text: str,
        *,
        progress_callback: Callable[[PostProcessProgress], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> PostProcessResult:
        del raw_text, progress_callback, cancel_token
        raise PostProcessUnavailableError("modele LLM local absent")


class RawOnlyPostProcessor:
    def build_outputs(
        self,
        raw_text: str,
        *,
        progress_callback: Callable[[PostProcessProgress], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> PostProcessResult:
        del progress_callback, cancel_token
        return PostProcessResult(outputs={MODE_RAW: raw_text.strip() + "\n"}, actions=[])


def fake_post_processor_factory(config: AppConfig) -> FakePostProcessor:
    del config
    return FakePostProcessor()


def failing_post_processor_factory(config: AppConfig) -> FailingPostProcessor:
    del config
    return FailingPostProcessor()


def raw_only_post_processor_factory(config: AppConfig) -> RawOnlyPostProcessor:
    del config
    return RawOnlyPostProcessor()


class PipelineTest(unittest.TestCase):
    def _config(self, tmpdir):
        return AppConfig(home=Path(tmpdir), allow_model_download=False)

    def _runner(self, cmd, check=False, capture_output=False, text=False):
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"fake wav")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "2.0\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def test_process_file_writes_raw_and_smart_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            source = config.inbox_dir / "audience.mp3"
            config.ensure_directories()
            source.write_bytes(b"audio")
            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=fake_engine_factory,
                post_processor_factory=fake_post_processor_factory,
                command_runner=self._runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready
            updates = []

            with patch("wisperauto.pipeline.shutil.which", lambda _cmd: "/usr/bin/tool"):
                record = pipeline.process_file(source, progress_callback=lambda item, _msg: updates.append(item))

            self.assertEqual(record.status, STATUS_DONE)
            self.assertIn(STATUS_POSTPROCESSING, {item.status for item in updates})
            self.assertTrue(Path(record.outputs[MODE_RAW]).exists())
            self.assertTrue(Path(record.outputs[MODE_SMART]).exists())
            self.assertIn("sauter ligne", Path(record.outputs[MODE_RAW]).read_text(encoding="utf-8"))
            self.assertIn("Bonjour, maitre.", Path(record.outputs[MODE_SMART]).read_text(encoding="utf-8"))
            self.assertFalse(source.exists())
            self.assertTrue(Path(record.processed_path).exists())

    def test_conversion_error_moves_source_to_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            source = config.inbox_dir / "corrompu.dss"
            config.ensure_directories()
            source.write_bytes(b"audio")

            def failing_runner(cmd, check=False, capture_output=False, text=False):
                if cmd[0] == "ffmpeg":
                    raise subprocess.CalledProcessError(1, cmd, stderr="fichier illisible")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=fake_engine_factory,
                command_runner=failing_runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready

            with patch("wisperauto.pipeline.shutil.which", lambda _cmd: "/usr/bin/tool"):
                record = pipeline.process_file(source)

            self.assertEqual(record.status, STATUS_ERROR)
            self.assertIn("fichier illisible", record.error)
            self.assertFalse(source.exists())
            self.assertTrue(Path(record.failed_path).exists())

    def test_import_audio_job_creates_ready_record_without_transcribing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            source = Path(tmpdir) / "source.mp3"
            config.ensure_directories()
            source.write_bytes(b"audio")
            pipeline = TranscriptionPipeline(config, JobStore(config.history_path))

            record = pipeline.import_audio_job(source)

            self.assertEqual(record.status, STATUS_READY)
            self.assertTrue(Path(record.source_path).exists())
            self.assertEqual(len(pipeline.store.latest()), 1)

    def test_cancelled_job_keeps_source_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            source = config.inbox_dir / "audience.mp3"
            config.ensure_directories()
            source.write_bytes(b"audio")
            token = CancellationToken()
            token.cancel()
            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=fake_engine_factory,
                post_processor_factory=fake_post_processor_factory,
                command_runner=self._runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready

            record = pipeline.process_file(source, cancel_token=token)

            self.assertEqual(record.status, STATUS_CANCELLED)
            self.assertTrue(source.exists())

    def test_engine_is_reused_across_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            config.ensure_directories()
            first = config.inbox_dir / "a.mp3"
            second = config.inbox_dir / "b.mp3"
            first.write_bytes(b"audio")
            second.write_bytes(b"audio")
            calls = []

            def engine_factory(config: AppConfig, allow_model_download: bool) -> FakeEngine:
                del config, allow_model_download
                calls.append("created")
                return FakeEngine()

            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=engine_factory,
                post_processor_factory=fake_post_processor_factory,
                command_runner=self._runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready

            with patch("wisperauto.pipeline.shutil.which", lambda _cmd: "/usr/bin/tool"):
                pipeline.process_file(first)
                pipeline.process_file(second)

            self.assertEqual(calls, ["created"])

    def test_post_processor_is_reused_across_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            config.ensure_directories()
            first = config.inbox_dir / "a.mp3"
            second = config.inbox_dir / "b.mp3"
            first.write_bytes(b"audio")
            second.write_bytes(b"audio")
            processors = []

            def post_processor_factory(config: AppConfig) -> FakePostProcessor:
                del config
                processor = FakePostProcessor()
                processors.append(processor)
                return processor

            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=fake_engine_factory,
                post_processor_factory=post_processor_factory,
                command_runner=self._runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready

            with patch("wisperauto.pipeline.shutil.which", lambda _cmd: "/usr/bin/tool"):
                pipeline.process_file(first)
                pipeline.process_file(second)

            self.assertEqual(len(processors), 1)
            self.assertEqual(processors[0].calls, 2)

    def test_postprocess_error_keeps_raw_output_and_source_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            source = config.inbox_dir / "audience.mp3"
            config.ensure_directories()
            source.write_bytes(b"audio")
            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=fake_engine_factory,
                post_processor_factory=failing_post_processor_factory,
                command_runner=self._runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready

            with patch("wisperauto.pipeline.shutil.which", lambda _cmd: "/usr/bin/tool"):
                record = pipeline.process_file(source)

            self.assertEqual(record.status, STATUS_ERROR)
            self.assertEqual(record.phase, "postprocess_error")
            self.assertIn("modele LLM local absent", record.error)
            self.assertTrue(source.exists())
            self.assertFalse(record.failed_path)
            self.assertTrue(Path(record.outputs[MODE_RAW]).exists())
            self.assertNotIn(MODE_SMART, record.outputs)

    def test_missing_selected_output_falls_back_to_available_raw_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), output_mode=MODE_REPORT)
            source = config.inbox_dir / "audience.mp3"
            config.ensure_directories()
            source.write_bytes(b"audio")
            pipeline = TranscriptionPipeline(
                config,
                JobStore(config.history_path),
                engine_factory=fake_engine_factory,
                post_processor_factory=raw_only_post_processor_factory,
                command_runner=self._runner,
            )
            pipeline.wait_until_file_ready = no_wait_until_file_ready

            with patch("wisperauto.pipeline.shutil.which", lambda _cmd: "/usr/bin/tool"):
                record = pipeline.process_file(source)

            self.assertEqual(record.status, STATUS_DONE)
            self.assertEqual(record.transcript_path, record.outputs[MODE_RAW])

    def test_safe_float_rejects_non_finite_values(self):
        self.assertIsNone(safe_float("nan"))
        self.assertIsNone(safe_float("inf"))
        self.assertEqual(safe_float("2.5"), 2.5)


if __name__ == "__main__":
    unittest.main()
