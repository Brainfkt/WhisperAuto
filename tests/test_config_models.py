import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wisperauto.config import (
    BACKEND_AUTO,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
    DEFAULT_MODEL_SIZE,
    DEFAULT_POSTPROCESS_MODEL_FILE,
    DEFAULT_POSTPROCESS_MODEL_REPO,
    POSTPROCESS_LLM_DIRECT,
    PROFILE_FAST,
    PROFILE_PRECISE,
    AppConfig,
    detect_existing_model_size,
    looks_like_model_dir,
)
from wisperauto.models import MLX_MODEL_REPOS, resolve_model_dir


class ConfigModelTest(unittest.TestCase):
    def test_default_model_is_turbo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))

            self.assertEqual(config.model_size, DEFAULT_MODEL_SIZE)
            self.assertEqual(config.postprocess_engine, POSTPROCESS_LLM_DIRECT)

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

            with patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.model_size, "medium")
            self.assertEqual(loaded.backend, BACKEND_WHISPER_CPP)
            self.assertEqual(loaded.output_mode, "raw")
            self.assertEqual(loaded.transcription_profile, PROFILE_PRECISE)

    def test_hugging_face_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            config = AppConfig(
                home=home,
                hf_token="hf_test_token",
                hf_fast_download=True,
                hf_xet_concurrency=24,
            )
            config.save_user_settings()

            with patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.hf_token, "hf_test_token")
            self.assertTrue(loaded.hf_fast_download)
            self.assertEqual(loaded.hf_xet_concurrency, 24)

    def test_postprocess_model_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            model_path = home / "models" / "postprocess" / "custom.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"gguf")
            config = AppConfig(
                home=home,
                postprocess_engine=POSTPROCESS_LLM_DIRECT,
                postprocess_model_path=str(model_path),
                postprocess_model_repo="repo/custom",
                postprocess_model_file="custom.gguf",
            )
            config.save_user_settings()

            with patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.postprocess_engine, POSTPROCESS_LLM_DIRECT)
            self.assertEqual(loaded.postprocess_model_repo, "repo/custom")
            self.assertEqual(loaded.postprocess_model_file, "custom.gguf")
            self.assertEqual(loaded.local_postprocess_model_path(), model_path)

    def test_legacy_cleaned_and_rules_settings_are_mapped_to_llm_smart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            home.mkdir(parents=True, exist_ok=True)
            (home / "settings.json").write_text(
                json.dumps({"output_mode": "cleaned", "postprocess_engine": "rules"}),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.output_mode, "smart")
            self.assertEqual(loaded.postprocess_engine, POSTPROCESS_LLM_DIRECT)

    def test_default_postprocess_model_path_uses_project_home(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            config.ensure_directories()
            model_path = config.postprocess_models_dir / DEFAULT_POSTPROCESS_MODEL_FILE
            model_path.write_bytes(b"gguf")

            self.assertEqual(config.postprocess_model_repo, DEFAULT_POSTPROCESS_MODEL_REPO)
            self.assertEqual(config.local_postprocess_model_path(), model_path)

    def test_existing_local_model_is_detected_before_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            medium = home / "models" / "medium"
            medium.mkdir(parents=True)
            (medium / "model.bin").write_bytes(b"model")

            self.assertEqual(detect_existing_model_size(home), "medium")
            with patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
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

    def test_mlx_model_repos_use_existing_mlx_repositories(self):
        self.assertEqual(MLX_MODEL_REPOS["small"], "mlx-community/whisper-small-mlx")
        self.assertEqual(MLX_MODEL_REPOS["medium"], "mlx-community/whisper-medium-mlx")
        self.assertEqual(MLX_MODEL_REPOS["large-v3-turbo"], "mlx-community/whisper-large-v3-turbo")
        self.assertEqual(MLX_MODEL_REPOS["large-v3"], "mlx-community/whisper-large-v3-mlx")

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
            cpp_model = cpp_dir / "ggml-large-v3-turbo.bin"
            cpp_model.write_bytes(b"model")

            self.assertEqual(config.local_model_path(BACKEND_MLX_WHISPER), mlx_dir)
            self.assertEqual(config.local_model_path(BACKEND_WHISPER_CPP), cpp_model)

    def test_whisper_cpp_model_must_match_current_model_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir), model_size="large-v3-turbo", backend=BACKEND_WHISPER_CPP)
            small_dir = config.models_dir / BACKEND_WHISPER_CPP / "small"
            small_dir.mkdir(parents=True)
            (small_dir / "ggml-small.bin").write_bytes(b"model")

            self.assertIsNone(config.local_model_path(BACKEND_WHISPER_CPP))

            current_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
            current_dir.mkdir(parents=True)
            current_model = current_dir / "ggml-large-v3-turbo.bin"
            current_model.write_bytes(b"model")

            self.assertEqual(config.local_model_path(BACKEND_WHISPER_CPP), current_model)

    def test_invalid_numeric_settings_do_not_crash_from_env_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            home.mkdir(parents=True, exist_ok=True)
            (home / "settings.json").write_text(
                json.dumps(
                    {
                        "backend": "not-a-backend",
                        "cpu_threads": "abc",
                        "batch_size": "-4",
                        "benchmark_seconds": "bad",
                        "transcription_profile": "unknown",
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"WISPERAUTO_HOME": str(home)}, clear=True):
                loaded = AppConfig.from_env()

            self.assertEqual(loaded.backend, BACKEND_AUTO)
            self.assertEqual(loaded.cpu_threads, 0)
            self.assertEqual(loaded.batch_size, 0)
            self.assertEqual(loaded.benchmark_seconds, 90)
            self.assertEqual(loaded.transcription_profile, PROFILE_FAST)


if __name__ == "__main__":
    unittest.main()
