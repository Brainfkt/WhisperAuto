"""Local backend benchmarking helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .backends import BACKEND_LABELS, BACKEND_FASTER_WHISPER, BACKEND_MLX_WHISPER, BACKEND_WHISPER_CPP, backend_health, create_engine
from .cancel import CancellationToken
from .config import AppConfig
from .errors import FFmpegUnavailableError


BenchmarkLogger = Callable[[str], None]


@dataclass(frozen=True)
class BenchmarkResult:
    backend_id: str
    backend_label: str
    model_size: str
    duration_seconds: float
    load_seconds: float
    transcribe_seconds: float
    realtime_factor: float
    segment_count: int
    character_count: int


def probe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def prepare_benchmark_audio(source_path: Path, target_path: Path, seconds: int) -> Path:
    if shutil.which("ffmpeg") is None:
        raise FFmpegUnavailableError()
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-t",
        str(max(10, seconds)),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return target_path


def ready_backend_ids(config: AppConfig) -> list[str]:
    candidates = [BACKEND_FASTER_WHISPER, BACKEND_MLX_WHISPER, BACKEND_WHISPER_CPP]
    return [backend for backend in candidates if backend_health(config, backend).ok_for_transcription]


def benchmark_backends(
    config: AppConfig,
    source_path: Path,
    logger: BenchmarkLogger | None = None,
    cancel_token: CancellationToken | None = None,
) -> list[BenchmarkResult]:
    logger = logger or (lambda _message: None)
    cancel_token = cancel_token or CancellationToken()
    source_path = Path(source_path)
    backends = ready_backend_ids(config)
    if not backends:
        raise RuntimeError("Aucun moteur avec modele local n'est pret pour le benchmark.")

    with tempfile.TemporaryDirectory(prefix="wisperauto-benchmark-") as tmpdir:
        clip_path = Path(tmpdir) / "benchmark.wav"
        logger(f"Preparation d'un extrait local de {config.benchmark_seconds}s.")
        prepare_benchmark_audio(source_path, clip_path, config.benchmark_seconds)
        duration = probe_duration(clip_path) or float(config.benchmark_seconds)
        results: list[BenchmarkResult] = []

        for backend_id in backends:
            cancel_token.raise_if_cancelled()
            bench_config = replace(config, backend=backend_id)
            label = BACKEND_LABELS.get(backend_id, backend_id)
            logger(f"Benchmark {label} avec modele {bench_config.model_size}.")

            load_started = time.monotonic()
            engine = create_engine(bench_config, allow_model_download=False)
            load_seconds = time.monotonic() - load_started
            cancel_token.raise_if_cancelled()

            transcribe_started = time.monotonic()
            if hasattr(engine, "transcribe_with_progress"):
                segments, _info = engine.transcribe_with_progress(clip_path, cancel_token=cancel_token)
            else:
                segments, _info = engine.transcribe(clip_path)
            segment_list = list(segments)
            transcribe_seconds = max(0.001, time.monotonic() - transcribe_started)
            text = "\n".join(getattr(segment, "text", "") for segment in segment_list)
            result = BenchmarkResult(
                backend_id=backend_id,
                backend_label=label,
                model_size=bench_config.model_size,
                duration_seconds=duration,
                load_seconds=load_seconds,
                transcribe_seconds=transcribe_seconds,
                realtime_factor=duration / transcribe_seconds,
                segment_count=len(segment_list),
                character_count=len(text),
            )
            logger(format_benchmark_result(result))
            results.append(result)

    return sorted(results, key=lambda item: item.realtime_factor, reverse=True)


def format_benchmark_result(result: BenchmarkResult) -> str:
    return (
        f"{result.backend_label} : chargement {result.load_seconds:.1f}s, "
        f"transcription {result.transcribe_seconds:.1f}s pour {result.duration_seconds:.1f}s audio, "
        f"{result.realtime_factor:.2f}x temps reel, {result.segment_count} segments."
    )


def recommendation_text(results: list[BenchmarkResult]) -> str:
    if not results:
        return "Aucun resultat de benchmark."
    best = results[0]
    return f"Moteur recommande pour ce poste et ce modele : {best.backend_label} ({best.realtime_factor:.2f}x temps reel)."
