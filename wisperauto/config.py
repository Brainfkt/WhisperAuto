"""Configuration and filesystem layout for WisperAuto."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "WisperAuto"
DEFAULT_MODEL_SIZE = "large-v3-turbo"
PROFILE_FAST = "fast"
PROFILE_BALANCED = "balanced"
PROFILE_PRECISE = "precise"
PROFILE_CHOICES = {
    PROFILE_FAST,
    PROFILE_BALANCED,
    PROFILE_PRECISE,
}
BACKEND_AUTO = "auto"
BACKEND_FASTER_WHISPER = "faster-whisper"
BACKEND_MLX_WHISPER = "mlx-whisper"
BACKEND_WHISPER_CPP = "whisper.cpp"
BACKEND_CHOICES = {
    BACKEND_AUTO,
    BACKEND_FASTER_WHISPER,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
}

SUPPORTED_EXTENSIONS = {
    ".ds2",
    ".dss",
    ".m4a",
    ".mp3",
    ".wav",
}

DIRECT_TRANSCRIBE_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
}


def model_dir_name(model_size: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_size).strip("-._") or "model"


def looks_like_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "model.bin").exists()


def looks_like_backend_model(path: Path, backend: str = BACKEND_FASTER_WHISPER) -> bool:
    backend = backend or BACKEND_FASTER_WHISPER
    if backend == BACKEND_WHISPER_CPP:
        if path.exists() and path.is_file() and path.suffix.lower() in {".bin", ".gguf"}:
            return True
        if path.exists() and path.is_dir():
            return any(item.is_file() and item.suffix.lower() in {".bin", ".gguf"} for item in path.iterdir())
        return False
    if backend == BACKEND_MLX_WHISPER:
        if not path.exists() or not path.is_dir():
            return False
        if not any(path.iterdir()):
            return False
        return any(
            item.name in {"config.json", "tokenizer.json"}
            or item.suffix.lower() in {".safetensors", ".npz"}
            for item in path.rglob("*")
            if item.is_file()
        )
    return looks_like_model_dir(path)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "oui", "on"}


def _safe_int(value, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return default
    return parsed


def _setting_int(settings: dict, key: str, env_name: str, default: int, minimum: int | None = None) -> int:
    return _safe_int(os.environ.get(env_name, settings.get(key, default)), default, minimum)


def _setting_choice(settings: dict, key: str, env_name: str, choices: set[str], default: str) -> str:
    value = str(os.environ.get(env_name, settings.get(key, default)) or default)
    return value if value in choices else default


def default_home() -> Path:
    configured = os.environ.get("WISPERAUTO_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Documents" / APP_NAME


def load_user_settings(home: Path) -> dict:
    settings_path = home / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def detect_existing_model_size(home: Path) -> str | None:
    models_dir = home / "models"
    if not models_dir.exists():
        return None
    for model_size in (DEFAULT_MODEL_SIZE, "medium", "small", "large-v3"):
        directory_name = model_dir_name(model_size)
        backend_candidate = models_dir / BACKEND_FASTER_WHISPER / directory_name
        if looks_like_model_dir(backend_candidate):
            return model_size
        if backend_candidate.exists() and any(backend_candidate.rglob("model.bin")):
            return model_size

        pointer_path = models_dir / f"{directory_name}.path"
        if pointer_path.exists():
            try:
                pointed_text = pointer_path.read_text(encoding="utf-8").strip()
            except OSError:
                pointed_text = ""
            if pointed_text and looks_like_model_dir(Path(pointed_text).expanduser()):
                return model_size

        candidate = models_dir / directory_name
        if looks_like_model_dir(candidate):
            return model_size
        if candidate.exists() and any(candidate.rglob("model.bin")):
            return model_size
    return None


@dataclass(frozen=True)
class AppConfig:
    home: Path
    model_size: str = DEFAULT_MODEL_SIZE
    language: str = "fr"
    compute_type: str = "int8"
    backend: str = BACKEND_AUTO
    device: str = "auto"
    whisper_cpp_binary: str = ""
    cpu_threads: int = 0
    num_workers: int = 1
    batch_size: int = 0
    whisper_cpp_threads: int = 0
    whisper_cpp_beam_size: int = 0
    whisper_cpp_best_of: int = 0
    vad_silence_ms: int = 0
    benchmark_seconds: int = 90
    max_file_mb: int = 2048
    max_duration_minutes: int = 240
    keep_intermediate_wav: bool = False
    allow_model_download: bool = False
    output_mode: str = "smart"
    transcription_profile: str = PROFILE_FAST
    model_download_timeout_minutes: int = 120
    disable_hf_xet: bool = True
    hf_token: str = ""
    hf_fast_download: bool = False
    hf_xet_concurrency: int = 32

    @classmethod
    def from_env(cls) -> "AppConfig":
        home = default_home()
        settings = load_user_settings(home)
        saved_model = str(settings.get("model_size") or "")
        model_size = os.environ.get("WISPERAUTO_MODEL") or saved_model or detect_existing_model_size(home) or DEFAULT_MODEL_SIZE
        saved_backend = str(settings.get("backend") or BACKEND_AUTO)
        backend = os.environ.get("WISPERAUTO_BACKEND") or saved_backend
        if backend not in BACKEND_CHOICES:
            backend = BACKEND_AUTO
        return cls(
            home=home,
            model_size=model_size,
            language=os.environ.get("WISPERAUTO_LANGUAGE", "fr"),
            compute_type=os.environ.get("WISPERAUTO_COMPUTE_TYPE", "int8"),
            backend=backend,
            device=os.environ.get("WISPERAUTO_DEVICE", str(settings.get("device") or "auto")),
            whisper_cpp_binary=os.environ.get(
                "WISPERAUTO_WHISPER_CPP_BINARY",
                str(settings.get("whisper_cpp_binary") or ""),
            ),
            cpu_threads=_setting_int(settings, "cpu_threads", "WISPERAUTO_CPU_THREADS", 0, minimum=0),
            num_workers=_setting_int(settings, "num_workers", "WISPERAUTO_NUM_WORKERS", 1, minimum=1),
            batch_size=_setting_int(settings, "batch_size", "WISPERAUTO_BATCH_SIZE", 0, minimum=0),
            whisper_cpp_threads=_setting_int(
                settings, "whisper_cpp_threads", "WISPERAUTO_WHISPER_CPP_THREADS", 0, minimum=0
            ),
            whisper_cpp_beam_size=_setting_int(
                settings, "whisper_cpp_beam_size", "WISPERAUTO_WHISPER_CPP_BEAM_SIZE", 0, minimum=0
            ),
            whisper_cpp_best_of=_setting_int(
                settings, "whisper_cpp_best_of", "WISPERAUTO_WHISPER_CPP_BEST_OF", 0, minimum=0
            ),
            vad_silence_ms=_setting_int(settings, "vad_silence_ms", "WISPERAUTO_VAD_SILENCE_MS", 0, minimum=0),
            benchmark_seconds=_setting_int(
                settings, "benchmark_seconds", "WISPERAUTO_BENCHMARK_SECONDS", 90, minimum=10
            ),
            max_file_mb=_safe_int(os.environ.get("WISPERAUTO_MAX_FILE_MB", "2048"), 2048, minimum=1),
            max_duration_minutes=_safe_int(
                os.environ.get("WISPERAUTO_MAX_DURATION_MINUTES", "240"), 240, minimum=1
            ),
            keep_intermediate_wav=_env_bool("WISPERAUTO_KEEP_WAV", False),
            allow_model_download=_env_bool("WISPERAUTO_ALLOW_MODEL_DOWNLOAD", False),
            output_mode=os.environ.get(
                "WISPERAUTO_OUTPUT_MODE",
                str(settings.get("output_mode") or "smart"),
            ),
            transcription_profile=_setting_choice(
                settings,
                "transcription_profile",
                "WISPERAUTO_TRANSCRIPTION_PROFILE",
                PROFILE_CHOICES,
                PROFILE_FAST,
            ),
            model_download_timeout_minutes=_safe_int(
                os.environ.get("WISPERAUTO_MODEL_DOWNLOAD_TIMEOUT_MINUTES", "120"), 120, minimum=1
            ),
            disable_hf_xet=_env_bool("WISPERAUTO_DISABLE_HF_XET", True),
            hf_token=os.environ.get("HF_TOKEN") or os.environ.get("WISPERAUTO_HF_TOKEN") or str(settings.get("hf_token") or ""),
            hf_fast_download=_env_bool(
                "WISPERAUTO_HF_FAST_DOWNLOAD",
                bool(settings.get("hf_fast_download") or False),
            ),
            hf_xet_concurrency=_setting_int(
                settings,
                "hf_xet_concurrency",
                "WISPERAUTO_HF_XET_CONCURRENCY",
                32,
                minimum=1,
            ),
        )

    @property
    def inbox_dir(self) -> Path:
        return self.home / "inbox"

    @property
    def outbox_dir(self) -> Path:
        return self.home / "outbox"

    @property
    def processed_dir(self) -> Path:
        return self.home / "processed"

    @property
    def failed_dir(self) -> Path:
        return self.home / "failed"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def model_dir(self) -> Path:
        return self.models_dir / model_dir_name(self.model_size)

    @property
    def model_pointer_path(self) -> Path:
        return self.models_dir / f"{model_dir_name(self.model_size)}.path"

    def backend_model_dir(self, backend: str | None = None) -> Path:
        backend = self._normal_backend(backend)
        return self.models_dir / backend / model_dir_name(self.model_size)

    def backend_model_pointer_path(self, backend: str | None = None) -> Path:
        backend = self._normal_backend(backend)
        return self.models_dir / backend / f"{model_dir_name(self.model_size)}.path"

    @property
    def history_path(self) -> Path:
        return self.logs_dir / "history.jsonl"

    @property
    def app_log_path(self) -> Path:
        return self.logs_dir / "wisperauto.log"

    @property
    def commands_path(self) -> Path:
        return self.home / "voice_commands.json"

    @property
    def settings_path(self) -> Path:
        return self.home / "settings.json"

    def ensure_directories(self) -> None:
        for folder in (
            self.inbox_dir,
            self.outbox_dir,
            self.processed_dir,
            self.failed_dir,
            self.logs_dir,
            self.models_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def local_model_path(self, backend: str | None = None) -> Path | None:
        backend = self._normal_backend(backend)
        env_name = "WISPERAUTO_MODEL_PATH"
        if backend == BACKEND_MLX_WHISPER:
            env_name = "WISPERAUTO_MLX_MODEL_PATH"
        elif backend == BACKEND_WHISPER_CPP:
            env_name = "WISPERAUTO_WHISPER_CPP_MODEL_PATH"

        configured = os.environ.get(env_name)
        if configured:
            path = Path(configured).expanduser()
            resolved = self._resolve_backend_model_path(path, backend)
            if resolved:
                return resolved

        pointer_path = self.backend_model_pointer_path(backend)
        if pointer_path.exists():
            try:
                pointed_text = pointer_path.read_text(encoding="utf-8").strip()
            except OSError:
                pointed_text = ""
            if pointed_text:
                resolved = self._resolve_backend_model_path(Path(pointed_text).expanduser(), backend)
                if resolved:
                    return resolved

        if backend == BACKEND_FASTER_WHISPER and self.model_pointer_path.exists():
            try:
                pointed_text = self.model_pointer_path.read_text(encoding="utf-8").strip()
            except OSError:
                pointed_text = ""
            if pointed_text:
                pointed = Path(pointed_text).expanduser()
                if looks_like_model_dir(pointed):
                    return pointed

        backend_dir = self.backend_model_dir(backend)
        resolved_backend_dir = self._resolve_backend_model_path(backend_dir, backend)
        if resolved_backend_dir:
            return resolved_backend_dir

        if backend == BACKEND_FASTER_WHISPER and looks_like_model_dir(self.model_dir):
            return self.model_dir

        if self.models_dir.exists():
            if backend == BACKEND_WHISPER_CPP:
                model_key = model_dir_name(self.model_size).lower()
                matches = sorted(
                    item
                    for suffix in ("*.bin", "*.gguf")
                    for item in self.models_dir.joinpath(backend).rglob(suffix)
                    if item.is_file()
                )
                for item in matches:
                    if model_key in str(item).lower():
                        return item
            elif backend == BACKEND_FASTER_WHISPER:
                matches = sorted(self.models_dir.rglob("model.bin"))
                for match in matches:
                    parent = match.parent
                    if model_dir_name(self.model_size).lower() in str(parent).lower():
                        return parent
        return None

    def remember_model_path(self, path: Path, backend: str | None = None) -> None:
        backend = self._normal_backend(backend)
        if not looks_like_backend_model(path, backend):
            raise ValueError(f"Dossier modele invalide : {path}")
        pointer_path = self.backend_model_pointer_path(backend)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(str(path), encoding="utf-8")
        if backend == BACKEND_FASTER_WHISPER:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.model_pointer_path.write_text(str(path), encoding="utf-8")

    def model_status_label(self, backend: str | None = None) -> str:
        backend = backend or (self.backend if self.backend != BACKEND_AUTO else BACKEND_FASTER_WHISPER)
        if self.local_model_path(backend):
            return f"Modele : {self.model_size} local"
        if self.allow_model_download:
            return f"Modele : {self.model_size} telechargeable"
        return f"Modele : {self.model_size} absent"

    def save_user_settings(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_size": self.model_size,
            "backend": self.backend,
            "device": self.device,
            "whisper_cpp_binary": self.whisper_cpp_binary,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "whisper_cpp_threads": self.whisper_cpp_threads,
            "whisper_cpp_beam_size": self.whisper_cpp_beam_size,
            "whisper_cpp_best_of": self.whisper_cpp_best_of,
            "vad_silence_ms": self.vad_silence_ms,
            "benchmark_seconds": self.benchmark_seconds,
            "output_mode": self.output_mode,
            "transcription_profile": self.transcription_profile,
            "hf_token": self.hf_token,
            "hf_fast_download": self.hf_fast_download,
            "hf_xet_concurrency": self.hf_xet_concurrency,
        }
        tmp_path = self.settings_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.settings_path)

    def _normal_backend(self, backend: str | None) -> str:
        backend = backend or self.backend or BACKEND_FASTER_WHISPER
        if backend == BACKEND_AUTO:
            backend = BACKEND_FASTER_WHISPER
        if backend not in BACKEND_CHOICES:
            return BACKEND_FASTER_WHISPER
        return backend

    def _resolve_backend_model_path(self, path: Path, backend: str) -> Path | None:
        if not path.exists():
            return None
        if backend == BACKEND_WHISPER_CPP:
            if path.is_file() and path.suffix.lower() in {".bin", ".gguf"}:
                return path
            if path.is_dir():
                matches = sorted(
                    item
                    for suffix in ("*.bin", "*.gguf")
                    for item in path.glob(suffix)
                    if item.is_file()
                )
                if matches:
                    return matches[0]
            return None
        if looks_like_backend_model(path, backend):
            return path
        if path.is_dir() and backend == BACKEND_FASTER_WHISPER:
            matches = sorted(path.rglob("model.bin"))
            for match in matches:
                parent = match.parent
                if model_dir_name(self.model_size).lower() in str(parent).lower() or path == self.backend_model_dir(backend):
                    return parent
            if matches:
                return matches[0].parent
        return None
