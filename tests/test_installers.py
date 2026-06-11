import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wisperauto.config import AppConfig
from wisperauto.config import BACKEND_FASTER_WHISPER, BACKEND_MLX_WHISPER, BACKEND_WHISPER_CPP
from wisperauto.installers import (
    InstallUnavailableError,
    backend_install_plan,
    download_model,
    faster_whisper_plan,
    ffmpeg_plan,
    hf_acceleration_plan,
    run_install_plan,
)


class InstallersTest(unittest.TestCase):
    def test_faster_whisper_plan_uses_current_python_and_requirements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("faster-whisper>=1.0,<2.0\n", encoding="utf-8")

            plan = faster_whisper_plan(root)

        self.assertEqual(plan.name, "faster-whisper")
        self.assertEqual(plan.commands[0][1:4], ["-m", "pip", "install"])
        self.assertTrue(str(plan.commands[1][-1]).endswith("requirements.txt"))

    def test_hf_acceleration_plan_installs_hub_and_xet(self):
        plan = hf_acceleration_plan()

        self.assertEqual(plan.name, "Accelerateur Hugging Face")
        self.assertIn("huggingface_hub", plan.commands[0])
        self.assertIn("hf-xet", plan.commands[0])

    def test_windows_ffmpeg_plan_prefers_winget(self):
        with patch("wisperauto.installers.sys.platform", "win32"):
            with patch("wisperauto.installers.command_exists", lambda name: name == "winget"):
                plan = ffmpeg_plan()

        self.assertEqual(plan.name, "FFmpeg")
        self.assertEqual(plan.commands[0][0], "winget")
        self.assertIn("Gyan.FFmpeg", plan.commands[0])

    def test_ffmpeg_plan_reports_missing_installer(self):
        with patch("wisperauto.installers.sys.platform", "win32"):
            with patch("wisperauto.installers.command_exists", lambda _name: False):
                with self.assertRaises(InstallUnavailableError):
                    ffmpeg_plan()

    def test_run_install_plan_logs_and_raises_on_failure(self):
        calls = []

        def runner(command, check=False, capture_output=False, text=False):
            calls.append(command)
            return subprocess.CompletedProcess(command, 1, "", "erreur")

        plan = faster_whisper_plan(Path("/tmp/absent"))
        with self.assertRaises(RuntimeError):
            run_install_plan(plan, logger=lambda _message: None, runner=runner)

        self.assertEqual(calls[0][1:4], ["-m", "pip", "install"])

    def test_download_model_records_local_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))

            def fake_run_streamed_command(command, logger, timeout_seconds=None, runner=None, env=None):
                del command, logger, timeout_seconds, runner, env
                target = config.model_dir / "snapshots" / "abc"
                target.mkdir(parents=True, exist_ok=True)
                (target / "model.bin").write_bytes(b"model")

            with patch("wisperauto.installers.run_streamed_command", fake_run_streamed_command):
                model_path = download_model(config, logger=lambda _message: None)

            self.assertEqual(model_path, config.model_dir / "snapshots" / "abc")
            self.assertEqual(config.local_model_path(), config.model_dir / "snapshots" / "abc")

    def test_download_model_fast_hf_uses_env_without_exposing_token_in_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                home=Path(tmpdir),
                hf_token="hf_secret",
                hf_fast_download=True,
                hf_xet_concurrency=40,
            )
            captured = {}

            def fake_run_streamed_command(command, logger, timeout_seconds=None, runner=None, env=None):
                del logger, timeout_seconds, runner
                captured["command"] = command
                captured["env"] = env or {}
                target = config.model_dir / "snapshots" / "abc"
                target.mkdir(parents=True, exist_ok=True)
                (target / "model.bin").write_bytes(b"model")

            with patch("wisperauto.installers.run_streamed_command", fake_run_streamed_command):
                download_model(config, logger=lambda _message: None)

            self.assertNotIn("hf_secret", " ".join(captured["command"]))
            self.assertEqual(captured["env"].get("HF_TOKEN"), "hf_secret")
            self.assertEqual(captured["env"].get("HF_XET_HIGH_PERFORMANCE"), "1")
            self.assertEqual(captured["env"].get("HF_XET_NUM_CONCURRENT_RANGE_GETS"), "40")
            self.assertIn("--hf-fast-download", captured["command"])

    def test_backend_install_plan_routes_to_mlx_on_apple_silicon(self):
        with patch("wisperauto.installers.sys.platform", "darwin"):
            with patch("wisperauto.installers.platform.machine", lambda: "arm64"):
                plan = backend_install_plan(BACKEND_MLX_WHISPER, Path("/tmp/absent"))

        self.assertEqual(plan.name, "mlx-whisper")
        self.assertIn("mlx-whisper>=0.4", plan.commands[-1])

    def test_backend_install_plan_routes_to_whisper_cpp_brew(self):
        with patch("wisperauto.installers.sys.platform", "darwin"):
            with patch("wisperauto.installers.command_exists", lambda name: name == "brew"):
                plan = backend_install_plan(BACKEND_WHISPER_CPP, Path("/tmp/absent"))

        self.assertEqual(plan.name, "whisper.cpp")
        self.assertEqual(plan.commands[0], ["brew", "install", "whisper-cpp"])

    def test_backend_install_plan_keeps_faster_whisper_default(self):
        plan = backend_install_plan(BACKEND_FASTER_WHISPER, Path("/tmp/absent"))

        self.assertEqual(plan.name, "faster-whisper")


if __name__ == "__main__":
    unittest.main()
