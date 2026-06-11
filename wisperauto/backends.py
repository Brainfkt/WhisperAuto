"""Transcription backend registry and engine adapters."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import (
    BACKEND_AUTO,
    BACKEND_FASTER_WHISPER,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
    PROFILE_BALANCED,
    PROFILE_FAST,
    PROFILE_PRECISE,
    AppConfig,
)
from .errors import DependencyUnavailableError, ModelUnavailableError
from .models import ensure_backend_model_available


BACKEND_LABELS = {
    BACKEND_AUTO: "Auto",
    BACKEND_FASTER_WHISPER: "faster-whisper",
    BACKEND_MLX_WHISPER: "MLX Mac",
    BACKEND_WHISPER_CPP: "whisper.cpp",
}

BACKEND_ORDER = (
    BACKEND_AUTO,
    BACKEND_FASTER_WHISPER,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
)

_MLX_RUNTIME_OK: bool | None = None
WHISPER_CPP_DIRECT_EXTENSIONS = {".flac", ".mp3", ".ogg", ".wav"}
ProgressUpdate = Callable[[float, str], None]


@dataclass(frozen=True)
class BackendHealth:
    backend_id: str
    label: str
    compatible: bool
    dependency_ok: bool
    model_local: bool
    model_path: Path | None
    device: str
    compute_type: str
    cpu_threads: int
    batch_size: int
    vad_silence_ms: int
    messages: list[str]

    @property
    def ok_for_transcription(self) -> bool:
        return self.compatible and self.dependency_ok and self.model_local


@dataclass
class TextSegment:
    text: str
    end: float | None = None


@dataclass
class TranscriptionInfo:
    duration: float | None = None


@dataclass(frozen=True)
class DecodeOptions:
    beam_size: int
    best_of: int
    condition_on_previous_text: bool
    batch_size: int
    cpu_threads: int
    num_workers: int
    vad_silence_ms: int
    whisper_cpp_threads: int
    whisper_cpp_beam_size: int
    whisper_cpp_best_of: int


def is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def cuda_available() -> bool:
    if platform.system() == "Darwin":
        return False
    if shutil.which("nvidia-smi") is not None:
        return True
    return False


def effective_cpu_threads(config: AppConfig) -> int:
    if config.cpu_threads > 0:
        return max(1, config.cpu_threads)
    cores = os.cpu_count() or 4
    return max(1, min(8, cores - 2 if cores > 4 else cores))


def effective_num_workers(config: AppConfig) -> int:
    if config.num_workers > 0:
        return max(1, config.num_workers)
    return 1


def effective_batch_size(config: AppConfig, backend_id: str, device: str) -> int:
    if config.batch_size > 0:
        return max(1, config.batch_size)
    if backend_id != BACKEND_FASTER_WHISPER:
        return 1
    if device == "cuda":
        return 8 if config.transcription_profile != PROFILE_PRECISE else 4
    if config.transcription_profile == PROFILE_FAST:
        return 4
    if config.transcription_profile == PROFILE_BALANCED:
        return 2
    return 1


def effective_vad_silence_ms(config: AppConfig) -> int:
    if config.vad_silence_ms > 0:
        return max(100, config.vad_silence_ms)
    if config.transcription_profile == PROFILE_FAST:
        return 500
    if config.transcription_profile == PROFILE_BALANCED:
        return 750
    return 1000


def decode_options(config: AppConfig, backend_id: str) -> DecodeOptions:
    device = backend_device(config, backend_id)
    if config.transcription_profile == PROFILE_PRECISE:
        beam_size = 5
        best_of = 5
        condition_on_previous_text = True
    elif config.transcription_profile == PROFILE_BALANCED:
        beam_size = 3
        best_of = 3
        condition_on_previous_text = True
    else:
        beam_size = 1
        best_of = 1
        condition_on_previous_text = False

    return DecodeOptions(
        beam_size=beam_size,
        best_of=best_of,
        condition_on_previous_text=condition_on_previous_text,
        batch_size=effective_batch_size(config, backend_id, device),
        cpu_threads=effective_cpu_threads(config),
        num_workers=effective_num_workers(config),
        vad_silence_ms=effective_vad_silence_ms(config),
        whisper_cpp_threads=max(1, config.whisper_cpp_threads or effective_cpu_threads(config)),
        whisper_cpp_beam_size=max(1, config.whisper_cpp_beam_size or beam_size),
        whisper_cpp_best_of=max(1, config.whisper_cpp_best_of or best_of),
    )


def python_dependency_available(module_name: str) -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec(module_name) is not None


def mlx_whisper_runtime_ok() -> bool:
    global _MLX_RUNTIME_OK
    if _MLX_RUNTIME_OK is not None:
        return _MLX_RUNTIME_OK
    if not is_macos_arm64() or not python_dependency_available("mlx_whisper"):
        _MLX_RUNTIME_OK = False
        return _MLX_RUNTIME_OK
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx_whisper"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        _MLX_RUNTIME_OK = False
    else:
        _MLX_RUNTIME_OK = result.returncode == 0
    return _MLX_RUNTIME_OK


def whisper_cpp_binary(config: AppConfig) -> Path | None:
    configured = config.whisper_cpp_binary.strip()
    candidates = [configured] if configured else []
    candidates.extend(["whisper-cli", "whisper-cpp", "whisper", "main"])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return path
        found = shutil.which(candidate)
        if found:
            return Path(found)
    return None


def resolve_backend_id(config: AppConfig) -> str:
    requested = config.backend or BACKEND_AUTO
    if requested != BACKEND_AUTO:
        return requested

    faster_ready = python_dependency_available("faster_whisper") and config.local_model_path(BACKEND_FASTER_WHISPER)
    cpp_ready = whisper_cpp_binary(config) is not None and config.local_model_path(BACKEND_WHISPER_CPP)
    mlx_ready = mlx_whisper_runtime_ok() and config.local_model_path(BACKEND_MLX_WHISPER)

    if is_macos_arm64():
        if config.transcription_profile == PROFILE_FAST and cpp_ready:
            return BACKEND_WHISPER_CPP
        if mlx_ready:
            return BACKEND_MLX_WHISPER
        return BACKEND_FASTER_WHISPER

    if cuda_available():
        return BACKEND_FASTER_WHISPER

    if config.transcription_profile == PROFILE_FAST and cpp_ready:
        return BACKEND_WHISPER_CPP
    if faster_ready:
        return BACKEND_FASTER_WHISPER
    if cpp_ready:
        return BACKEND_WHISPER_CPP
    return BACKEND_FASTER_WHISPER


def backend_device(config: AppConfig, backend_id: str) -> str:
    configured = (config.device or "auto").lower()
    if configured != "auto":
        return configured
    if backend_id == BACKEND_FASTER_WHISPER and cuda_available():
        return "cuda"
    if backend_id == BACKEND_MLX_WHISPER:
        return "mlx"
    if backend_id == BACKEND_WHISPER_CPP:
        return "local"
    return "cpu"


def backend_compute_type(config: AppConfig, backend_id: str, device: str) -> str:
    if backend_id == BACKEND_FASTER_WHISPER:
        if device == "cuda" and config.compute_type == "int8":
            return "float16"
        return config.compute_type
    if backend_id == BACKEND_MLX_WHISPER:
        return "mlx"
    if backend_id == BACKEND_WHISPER_CPP:
        return "quantized"
    return config.compute_type


def backend_dependency_ok(config: AppConfig, backend_id: str) -> bool:
    if backend_id == BACKEND_FASTER_WHISPER:
        return python_dependency_available("faster_whisper")
    if backend_id == BACKEND_MLX_WHISPER:
        return mlx_whisper_runtime_ok()
    if backend_id == BACKEND_WHISPER_CPP:
        return whisper_cpp_binary(config) is not None
    return False


def backend_compatible(backend_id: str) -> bool:
    if backend_id == BACKEND_MLX_WHISPER:
        return is_macos_arm64()
    return backend_id in {BACKEND_FASTER_WHISPER, BACKEND_WHISPER_CPP}


def backend_health(config: AppConfig, backend_id: str | None = None) -> BackendHealth:
    resolved = backend_id or resolve_backend_id(config)
    compatible = backend_compatible(resolved)
    dependency_ok = backend_dependency_ok(config, resolved)
    model_path = config.local_model_path(resolved)
    device = backend_device(config, resolved)
    compute_type = backend_compute_type(config, resolved, device)
    options = decode_options(config, resolved)
    messages: list[str] = []

    if not compatible:
        messages.append(f"{BACKEND_LABELS.get(resolved, resolved)} n'est pas compatible avec cet ordinateur.")
    elif not dependency_ok:
        if resolved == BACKEND_FASTER_WHISPER:
            messages.append("La dependance Python faster-whisper est absente.")
        elif resolved == BACKEND_MLX_WHISPER:
            messages.append("mlx-whisper est absent ou cet ordinateur n'est pas un Mac Apple Silicon.")
        elif resolved == BACKEND_WHISPER_CPP:
            messages.append("Le binaire whisper.cpp est introuvable.")
    if not model_path:
        messages.append(f"Aucun modele local {BACKEND_LABELS.get(resolved, resolved)} n'a ete trouve.")

    return BackendHealth(
        backend_id=resolved,
        label=BACKEND_LABELS.get(resolved, resolved),
        compatible=compatible,
        dependency_ok=dependency_ok,
        model_local=model_path is not None,
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=options.cpu_threads if resolved == BACKEND_FASTER_WHISPER else options.whisper_cpp_threads,
        batch_size=options.batch_size,
        vad_silence_ms=options.vad_silence_ms,
        messages=messages,
    )


class FasterWhisperEngine:
    def __init__(self, config: AppConfig, allow_model_download: bool = False):
        if not python_dependency_available("faster_whisper"):
            raise DependencyUnavailableError("faster-whisper")

        from faster_whisper import WhisperModel

        local_model = config.local_model_path(BACKEND_FASTER_WHISPER)
        if local_model:
            model_source = str(local_model)
        elif allow_model_download or config.allow_model_download:
            model_source = str(ensure_backend_model_available(config, BACKEND_FASTER_WHISPER))
        else:
            raise ModelUnavailableError()

        self.config = config
        self.device = backend_device(config, BACKEND_FASTER_WHISPER)
        self.compute_type = backend_compute_type(config, BACKEND_FASTER_WHISPER, self.device)
        self.options = decode_options(config, BACKEND_FASTER_WHISPER)
        kwargs: dict[str, Any] = {
            "compute_type": self.compute_type,
            "cpu_threads": self.options.cpu_threads,
            "num_workers": self.options.num_workers,
        }
        if self.device in {"cpu", "cuda"}:
            kwargs["device"] = self.device
        self.model = WhisperModel(model_source, **kwargs)
        self.batched_model = None
        if self.options.batch_size > 1:
            from faster_whisper import BatchedInferencePipeline

            self.batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe(self, audio_path: Path):
        options = {
            "language": self.config.language,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": self.options.vad_silence_ms},
            "beam_size": self.options.beam_size,
            "best_of": self.options.best_of,
            "condition_on_previous_text": self.options.condition_on_previous_text,
            "without_timestamps": False,
        }
        if self.batched_model is not None:
            options["batch_size"] = self.options.batch_size
            return self.batched_model.transcribe(str(audio_path), **options)
        return self.model.transcribe(str(audio_path), **options)


class MlxWhisperEngine:
    def __init__(self, config: AppConfig, allow_model_download: bool = False):
        if not is_macos_arm64() or not python_dependency_available("mlx_whisper"):
            raise DependencyUnavailableError("mlx-whisper")

        local_model = config.local_model_path(BACKEND_MLX_WHISPER)
        if local_model:
            self.model_source = str(local_model)
        elif allow_model_download or config.allow_model_download:
            self.model_source = str(ensure_backend_model_available(config, BACKEND_MLX_WHISPER))
        else:
            raise ModelUnavailableError()
        self.config = config

    def transcribe(self, audio_path: Path):
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_source,
            language=self.config.language,
            verbose=False,
        )
        return normalize_dict_result(result)


class WhisperCppEngine:
    def __init__(self, config: AppConfig, allow_model_download: bool = False):
        binary = whisper_cpp_binary(config)
        if not binary:
            raise DependencyUnavailableError("whisper.cpp")

        local_model = config.local_model_path(BACKEND_WHISPER_CPP)
        if local_model:
            self.model_path = local_model
        elif allow_model_download or config.allow_model_download:
            self.model_path = ensure_backend_model_available(config, BACKEND_WHISPER_CPP)
        else:
            raise ModelUnavailableError()

        self.binary = binary
        self.config = config
        self.options = decode_options(config, BACKEND_WHISPER_CPP)

    def transcribe(self, audio_path: Path):
        return self.transcribe_with_progress(audio_path)

    def transcribe_with_progress(
        self,
        audio_path: Path,
        progress_update: ProgressUpdate | None = None,
        cancel_token=None,
    ):
        with tempfile.TemporaryDirectory(prefix="wisperauto-whispercpp-") as tmpdir:
            output_prefix = Path(tmpdir) / "transcript"
            command = [
                str(self.binary),
                "-m",
                str(self.model_path),
                "-f",
                str(audio_path),
                "-l",
                self.config.language,
                "-t",
                str(self.options.whisper_cpp_threads),
                "-bs",
                str(self.options.whisper_cpp_beam_size),
                "-bo",
                str(self.options.whisper_cpp_best_of),
                "-oj",
                "--print-progress",
                "-of",
                str(output_prefix),
            ]
            output_lines: list[str] = []
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            lines: queue.Queue[str] = queue.Queue()

            def read_output() -> None:
                try:
                    for item in process.stdout:
                        lines.put(item)
                finally:
                    lines.put("")

            threading.Thread(target=read_output, daemon=True).start()
            output_done = False
            try:
                while True:
                    if cancel_token is not None:
                        cancel_token.raise_if_cancelled()
                    try:
                        while True:
                            line = lines.get_nowait()
                            if line == "":
                                output_done = True
                                break
                            cleaned = line.strip()
                            if cleaned:
                                output_lines.append(cleaned)
                                progress = parse_whisper_cpp_progress(cleaned)
                                if progress is not None and progress_update is not None:
                                    progress_update(progress / 100.0, f"whisper.cpp {progress}%")
                    except queue.Empty:
                        pass
                    returncode = process.poll()
                    if returncode is not None and output_done:
                        break
                    time.sleep(0.1)
            except Exception:
                terminate_process(process)
                raise

            if returncode != 0:
                details = "\n".join(output_lines[-20:]) or "whisper.cpp a echoue"
                raise RuntimeError(details)

            json_path = output_prefix.with_suffix(".json")
            if not json_path.exists():
                raise RuntimeError("whisper.cpp n'a pas produit de sortie JSON exploitable.")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            return normalize_dict_result(payload)


def normalize_dict_result(payload: dict) -> tuple[list[TextSegment], TranscriptionInfo]:
    raw_segments = payload.get("segments") if isinstance(payload, dict) else None
    if raw_segments is None and isinstance(payload, dict):
        raw_segments = payload.get("transcription")
    segments: list[TextSegment] = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            end = extract_segment_end(item)
            segments.append(TextSegment(text=text, end=float(end) if isinstance(end, (int, float)) else None))

    if not segments and isinstance(payload, dict):
        text = str(payload.get("text") or "").strip()
        if text:
            segments.append(TextSegment(text=text, end=None))

    duration = None
    if segments:
        known_ends = [segment.end for segment in segments if segment.end is not None]
        if known_ends:
            duration = max(known_ends)
    return segments, TranscriptionInfo(duration=duration)


def parse_whisper_cpp_progress(line: str) -> int | None:
    match = re.search(r"progress\s*=\s*(\d{1,3})%", line, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(\d{1,3})%\b", line)
    if not match:
        return None
    value = int(match.group(1))
    if 0 <= value <= 100:
        return value
    return None


def terminate_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def extract_segment_end(item: dict) -> float | None:
    end = item.get("end")
    if isinstance(end, (int, float)):
        return float(end)
    if isinstance(end, str):
        parsed = parse_float_or_timestamp(end)
        if parsed is not None:
            return parsed

    offsets = item.get("offsets")
    if isinstance(offsets, dict):
        offset_end = offsets.get("to")
        if isinstance(offset_end, (int, float)):
            return float(offset_end) / 1000.0
        if isinstance(offset_end, str):
            parsed = parse_float_or_timestamp(offset_end)
            if parsed is not None:
                return parsed if ":" in offset_end else parsed / 1000.0

    timestamps = item.get("timestamps")
    if isinstance(timestamps, dict):
        timestamp_end = timestamps.get("to")
        if isinstance(timestamp_end, str):
            return parse_float_or_timestamp(timestamp_end)
    return None


def parse_float_or_timestamp(value: str) -> float | None:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass

    parts = value.replace(",", ".").split(":")
    if not parts:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    total = 0.0
    for number in numbers:
        total = total * 60 + number
    return total


def create_engine(config: AppConfig, allow_model_download: bool = False):
    backend_id = resolve_backend_id(config)
    if backend_id == BACKEND_MLX_WHISPER:
        return MlxWhisperEngine(config, allow_model_download)
    if backend_id == BACKEND_WHISPER_CPP:
        return WhisperCppEngine(config, allow_model_download)
    return FasterWhisperEngine(config, allow_model_download)
