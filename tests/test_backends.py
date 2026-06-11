import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wisperauto.backends import backend_health, decode_options, normalize_dict_result, parse_whisper_cpp_progress, resolve_backend_id
from wisperauto.config import (
    BACKEND_AUTO,
    BACKEND_FASTER_WHISPER,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
    AppConfig,
)


class BackendsTest(unittest.TestCase):
    def test_auto_prefers_mlx_on_mac_arm64_when_installed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_AUTO)
            model_dir = config.backend_model_dir(BACKEND_MLX_WHISPER)
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with patch("wisperauto.backends.platform.system", lambda: "Darwin"):
                with patch("wisperauto.backends.platform.machine", lambda: "arm64"):
                    with patch("wisperauto.backends.mlx_whisper_runtime_ok", lambda: True):
                        self.assertEqual(resolve_backend_id(config), BACKEND_MLX_WHISPER)

    def test_auto_fast_prefers_whisper_cpp_on_mac_when_model_is_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_AUTO)
            model_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
            model_dir.mkdir(parents=True)
            (model_dir / "ggml-large-v3-turbo.bin").write_bytes(b"model")
            binary = Path(tmpdir) / "whisper-cli"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            with patch("wisperauto.backends.platform.system", lambda: "Darwin"):
                with patch("wisperauto.backends.platform.machine", lambda: "arm64"):
                    with patch("wisperauto.backends.mlx_whisper_runtime_ok", lambda: False):
                        with patch("wisperauto.backends.whisper_cpp_binary", lambda _config: binary):
                            self.assertEqual(resolve_backend_id(config), BACKEND_WHISPER_CPP)

    def test_auto_falls_back_to_faster_whisper_on_mac_without_mlx_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_AUTO)

            with patch("wisperauto.backends.platform.system", lambda: "Darwin"):
                with patch("wisperauto.backends.platform.machine", lambda: "arm64"):
                    with patch("wisperauto.backends.mlx_whisper_runtime_ok", lambda: False):
                        self.assertEqual(resolve_backend_id(config), BACKEND_FASTER_WHISPER)

    def test_auto_prefers_mlx_over_whisper_cpp_on_mac_when_both_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_AUTO)
            mlx_dir = config.backend_model_dir(BACKEND_MLX_WHISPER)
            mlx_dir.mkdir(parents=True)
            (mlx_dir / "config.json").write_text("{}", encoding="utf-8")
            cpp_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
            cpp_dir.mkdir(parents=True)
            (cpp_dir / "ggml-large-v3-turbo.bin").write_bytes(b"model")
            binary = Path(tmpdir) / "whisper-cli"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            with patch("wisperauto.backends.platform.system", lambda: "Darwin"):
                with patch("wisperauto.backends.platform.machine", lambda: "arm64"):
                    with patch("wisperauto.backends.mlx_whisper_runtime_ok", lambda: True):
                        with patch("wisperauto.backends.whisper_cpp_binary", lambda _config: binary):
                            self.assertEqual(resolve_backend_id(config), BACKEND_MLX_WHISPER)

    def test_auto_prefers_faster_whisper_over_whisper_cpp_on_mac_without_mlx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_AUTO)
            faster_dir = config.backend_model_dir(BACKEND_FASTER_WHISPER)
            faster_dir.mkdir(parents=True)
            (faster_dir / "model.bin").write_bytes(b"model")
            cpp_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
            cpp_dir.mkdir(parents=True)
            (cpp_dir / "ggml-large-v3-turbo.bin").write_bytes(b"model")
            binary = Path(tmpdir) / "whisper-cli"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            with patch("wisperauto.backends.platform.system", lambda: "Darwin"):
                with patch("wisperauto.backends.platform.machine", lambda: "arm64"):
                    with patch("wisperauto.backends.mlx_whisper_runtime_ok", lambda: False):
                        with patch("wisperauto.backends.whisper_cpp_binary", lambda _config: binary):
                            with patch(
                                "wisperauto.backends.python_dependency_available",
                                lambda name: name == "faster_whisper",
                            ):
                                self.assertEqual(resolve_backend_id(config), BACKEND_FASTER_WHISPER)

    def test_faster_whisper_uses_cuda_policy_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_FASTER_WHISPER)
            model_dir = config.backend_model_dir(BACKEND_FASTER_WHISPER)
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_bytes(b"model")

            with patch("wisperauto.backends.cuda_available", lambda: True):
                with patch(
                    "wisperauto.backends.python_dependency_available",
                    lambda name: name == "faster_whisper",
                ):
                    health = backend_health(config)

            self.assertTrue(health.ok_for_transcription)
            self.assertEqual(health.device, "cuda")
            self.assertEqual(health.compute_type, "float16")

    def test_whisper_cpp_health_uses_configured_binary_and_model_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), backend=BACKEND_WHISPER_CPP)
            model_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
            model_dir.mkdir(parents=True)
            model_file = model_dir / "ggml-large-v3-turbo.bin"
            model_file.write_bytes(b"model")
            binary = Path(tmpdir) / "whisper-cli"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            with patch("wisperauto.backends.whisper_cpp_binary", lambda _config: binary):
                health = backend_health(config)

            self.assertTrue(health.ok_for_transcription)
            self.assertEqual(health.model_path, model_file)

    def test_whisper_cpp_json_transcription_shape_is_normalized(self):
        payload = {
            "transcription": [
                {
                    "text": "Bonjour maitre",
                    "offsets": {"from": 0, "to": 2500},
                    "timestamps": {"from": "00:00:00.000", "to": "00:00:02.500"},
                }
            ]
        }

        segments, info = normalize_dict_result(payload)

        self.assertEqual(segments[0].text, "Bonjour maitre")
        self.assertEqual(segments[0].end, 2.5)
        self.assertEqual(info.duration, 2.5)

    def test_fast_profile_uses_batch_and_cpu_thread_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))

            with patch("wisperauto.backends.os.cpu_count", lambda: 8):
                options = decode_options(config, BACKEND_FASTER_WHISPER)

            self.assertEqual(options.cpu_threads, 6)
            self.assertEqual(options.batch_size, 4)
            self.assertEqual(options.beam_size, 1)
            self.assertEqual(options.vad_silence_ms, 500)

    def test_whisper_cpp_progress_line_is_parsed(self):
        self.assertEqual(parse_whisper_cpp_progress("whisper_print_progress_callback: progress =  42%"), 42)
        self.assertIsNone(parse_whisper_cpp_progress("no progress here"))


if __name__ == "__main__":
    unittest.main()
