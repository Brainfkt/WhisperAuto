import tempfile
import unittest
from pathlib import Path

from wisperauto.config import (
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
    DEFAULT_MODEL_SIZE,
    PROFILE_PRECISE,
    AppConfig,
    detect_existing_model_size,
    looks_like_model_dir,
)
from wisperauto.models import resolve_model_dir


class ConfigModelTest(unittest.TestCase):
    def test_default_model_is_turbo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))

            self.assertEqual(config.model_size, DEFAULT_MODEL_SIZE)

    def test_save_user_settings_round_trips_from_env_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            config = AppConfig(
                home=home,
                model_size="medium",
                backend=BACKEND_WHISPER_CPP,
                output_mode="raw",
                transcription_profile=PROFILE_PRECISE,
            )
            config.save_user_settings()

            with unittest.mock.patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.model_size, "medium")
            self.assertEqual(loaded.backend, BACKEND_WHISPER_CPP)
            self.assertEqual(loaded.output_mode, "raw")
            self.assertEqual(loaded.transcription_profile, PROFILE_PRECISE)

    def test_existing_local_model_is_detected_before_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            medium = home / "models" / "medium"
            medium.mkdir(parents=True)
            (medium / "model.bin").write_bytes(b"model")

            self.assertEqual(detect_existing_model_size(home), "medium")
            with unittest.mock.patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.model_size, "medium")

    def test_empty_model_pointer_does_not_resolve_to_current_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            config.ensure_directories()
            config.model_pointer_path.write_text("", encoding="utf-8")

            self.assertIsNone(config.local_model_path())

    def test_model_pointer_must_point_to_valid_model_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            config.ensure_directories()
            invalid = config.models_dir / "large-v3-invalid"
            invalid.mkdir()
            config.model_pointer_path.write_text(str(invalid), encoding="utf-8")

            self.assertIsNone(config.local_model_path())

    def test_valid_model_directory_requires_model_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(looks_like_model_dir(model_dir))

            (model_dir / "model.bin").write_bytes(b"model")
            self.assertTrue(looks_like_model_dir(model_dir))

    def test_resolve_model_dir_finds_nested_model_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            nested = config.model_dir / "snapshots" / "abc"
            nested.mkdir(parents=True)
            (nested / "model.bin").write_bytes(b"model")

            self.assertEqual(resolve_model_dir(config.model_dir, config), nested)

    def test_backend_model_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            mlx_dir = config.backend_model_dir(BACKEND_MLX_WHISPER)
            cpp_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
            mlx_dir.mkdir(parents=True)
            cpp_dir.mkdir(parents=True)
            (mlx_dir / "config.json").write_text("{}", encoding="utf-8")
            cpp_model = cpp_dir / "ggml-small.bin"
            cpp_model.write_bytes(b"model")

            self.assertEqual(config.local_model_path(BACKEND_MLX_WHISPER), mlx_dir)
            self.assertEqual(config.local_model_path(BACKEND_WHISPER_CPP), cpp_model)


if __name__ == "__main__":
    unittest.main()
