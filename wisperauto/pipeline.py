"""Reliable local audio transcription pipeline."""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from .cancel import CancellationToken
from .backends import (
    BACKEND_LABELS,
    WHISPER_CPP_DIRECT_EXTENSIONS,
    backend_dependency_ok,
    backend_health,
    create_engine,
    resolve_backend_id,
)
from .config import (
    BACKEND_FASTER_WHISPER,
    BACKEND_WHISPER_CPP,
    DIRECT_TRANSCRIBE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    AppConfig,
)
from .errors import (
    AudioFileTooLargeError,
    AudioTooLongError,
    EmptyAudioFileError,
    FFmpegConversionError,
    FFmpegUnavailableError,
    FileNotReadyError,
    OperationCancelledError,
    UnsupportedFormatError,
    WisperAutoError,
)
from .jobs import (
    STATUS_CANCELLED,
    STATUS_CONVERTING,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_READY,
    STATUS_TRANSCRIBING,
    JobRecord,
    JobStore,
    utc_now,
)
from .postprocess import MODE_SMART, PostProcessor


ProgressCallback = Callable[[JobRecord, str], None]


class Engine(Protocol):
    def transcribe(self, audio_path: Path):
        ...


class EngineFactory(Protocol):
    def __call__(self, config: AppConfig, allow_model_download: bool) -> Engine:
        ...


@dataclass
class PreflightReport:
    ffmpeg_ok: bool
    ffprobe_ok: bool
    faster_whisper_ok: bool
    model_local: bool
    backend_id: str
    backend_label: str
    backend_dependency_ok: bool
    backend_compatible: bool
    device: str
    compute_type: str
    cpu_threads: int
    batch_size: int
    vad_silence_ms: int
    messages: list[str]

    @property
    def ok_for_local_run(self) -> bool:
        return self.backend_compatible and self.backend_dependency_ok and self.model_local


def sanitize_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "audio"


def make_job_id(path: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{sanitize_stem(path.stem)}"


def safe_destination(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"reste environ {seconds}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"reste environ {minutes}min {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"reste environ {hours}h {minutes:02d}min"


def check_environment(config: AppConfig) -> PreflightReport:
    messages: list[str] = []
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    faster_whisper_ok = backend_dependency_ok(config, BACKEND_FASTER_WHISPER)
    health = backend_health(config)
    model_local = health.model_local

    messages.extend(health.messages)
    if not ffmpeg_ok:
        messages.append("FFmpeg est introuvable ; il reste requis pour convertir certains formats.")
    if not ffprobe_ok:
        messages.append("ffprobe est introuvable ; les durees et ETA seront moins precises.")

    return PreflightReport(
        ffmpeg_ok=ffmpeg_ok,
        ffprobe_ok=ffprobe_ok,
        faster_whisper_ok=faster_whisper_ok,
        model_local=model_local,
        backend_id=health.backend_id,
        backend_label=health.label,
        backend_dependency_ok=health.dependency_ok,
        backend_compatible=health.compatible,
        device=health.device,
        compute_type=health.compute_type,
        cpu_threads=health.cpu_threads,
        batch_size=health.batch_size,
        vad_silence_ms=health.vad_silence_ms,
        messages=messages,
    )


class TranscriptionPipeline:
    def __init__(
        self,
        config: AppConfig,
        store: JobStore | None = None,
        engine_factory: EngineFactory | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ):
        self.config = config
        self.config.ensure_directories()
        self.store = store or JobStore(config.history_path)
        self.engine_factory = engine_factory or create_engine
        self.command_runner = command_runner or subprocess.run
        self._engine: Engine | None = None
        self._engine_key: tuple[object, ...] | None = None
        self._engine_lock = threading.Lock()

    def reset_engine(self) -> None:
        with self._engine_lock:
            self._engine = None
            self._engine_key = None

    def import_audio_file(self, source_path: Path) -> Path:
        source_path = Path(source_path)
        self.validate_audio_file(source_path)
        destination = safe_destination(self.config.inbox_dir, source_path.name)
        shutil.copy2(source_path, destination)
        return destination

    def import_audio_job(self, source_path: Path) -> JobRecord:
        destination = self.import_audio_file(source_path)
        record = JobRecord(
            id=make_job_id(destination),
            source_name=destination.name,
            source_path=str(destination),
            status=STATUS_READY,
            progress=0,
            phase="ready",
            message="Fichier pret a transcrire.",
            model_size=self.config.model_size,
            backend=resolve_backend_id(self.config),
            transcription_profile=self.config.transcription_profile,
        )
        self.store.append(record)
        return record

    def process_file(
        self,
        source_path: Path,
        allow_model_download: bool = False,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> JobRecord:
        source_path = Path(source_path)
        record = JobRecord(
            id=make_job_id(source_path),
            source_name=source_path.name,
            source_path=str(source_path),
            status=STATUS_READY,
            progress=0,
            phase="ready",
            message="Fichier pret a transcrire.",
            model_size=self.config.model_size,
            backend=resolve_backend_id(self.config),
            transcription_profile=self.config.transcription_profile,
        )
        self.store.append(record)
        return self.process_record(record, allow_model_download, progress_callback, cancel_token)

    def process_record(
        self,
        record: JobRecord,
        allow_model_download: bool = False,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> JobRecord:
        cancel_token = cancel_token or CancellationToken()
        source_path = Path(record.source_path)
        temp_wav_path: Path | None = None
        created_temp_wav = False

        try:
            cancel_token.raise_if_cancelled()
            record = self._update(
                record,
                progress_callback,
                status=STATUS_QUEUED,
                progress=2,
                phase="queued",
                message="Fichier ajoute a la file.",
                error="",
                started_at=utc_now(),
                finished_at="",
                model_size=self.config.model_size,
                backend=resolve_backend_id(self.config),
                transcription_profile=self.config.transcription_profile,
            )

            self._wait_until_file_ready(source_path, cancel_token)
            self.validate_audio_file(source_path)

            duration = self.probe_duration(source_path)
            if duration is not None:
                record = self._update(
                    record,
                    progress_callback,
                    progress=5,
                    phase="probe",
                    message="Duree audio detectee.",
                    duration_seconds=duration,
                    progress_detail=format_duration(duration),
                )
                max_seconds = self.config.max_duration_minutes * 60
                if duration > max_seconds:
                    raise AudioTooLongError(duration / 60, self.config.max_duration_minutes)
            else:
                record = self._update(
                    record,
                    progress_callback,
                    progress=5,
                    phase="probe",
                    message="Duree audio non detectee ; estimation indisponible.",
                    progress_detail="ETA indisponible",
                )

            audio_path = source_path
            if self.needs_conversion(source_path):
                record = self._update(
                    record,
                    progress_callback,
                    status=STATUS_CONVERTING,
                    progress=8,
                    phase="converting",
                    message="Conversion audio en cours.",
                    progress_detail="Preparation FFmpeg",
                )
                temp_wav_path = self.convert_to_wav(
                    source_path,
                    record,
                    duration=duration,
                    progress_callback=progress_callback,
                    cancel_token=cancel_token,
                )
                created_temp_wav = True
                audio_path = temp_wav_path
                if duration is None:
                    duration = self.probe_duration(audio_path)
                    if duration is not None:
                        record = self._update(
                            record,
                            progress_callback,
                            duration_seconds=duration,
                            progress_detail=format_duration(duration),
                        )
            else:
                record = self._update(
                    record,
                    progress_callback,
                    status=STATUS_CONVERTING,
                    progress=18,
                    phase="prepare_audio",
                    message="Format lu directement, conversion evitee.",
                    progress_detail="MP3/M4A/WAV pris en charge directement",
                )

            cancel_token.raise_if_cancelled()
            record = self._update(
                record,
                progress_callback,
                status=STATUS_TRANSCRIBING,
                progress=20,
                phase="loading_model",
                message=f"Chargement du moteur {BACKEND_LABELS.get(resolve_backend_id(self.config), resolve_backend_id(self.config))}.",
                progress_detail="Modele conserve en memoire pour le batch",
            )
            txt_path = self.transcribe_audio(
                audio_path,
                record,
                allow_model_download=allow_model_download,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
            )

            processed_path = safe_destination(self.config.processed_dir, source_path.name)
            shutil.move(str(source_path), processed_path)
            if created_temp_wav and temp_wav_path and temp_wav_path.exists() and not self.config.keep_intermediate_wav:
                temp_wav_path.unlink()

            record = self._update(
                record,
                progress_callback,
                status=STATUS_DONE,
                progress=100,
                phase="done",
                message="Transcription terminee.",
                progress_detail="Pret pour copie ou export",
                eta_seconds=0,
                transcript_path=str(txt_path),
                processed_path=str(processed_path),
                error="",
                finished_at=utc_now(),
            )
            return record
        except OperationCancelledError:
            record = self._update(
                record,
                progress_callback,
                status=STATUS_CANCELLED,
                phase="cancelled",
                message="Operation annulee.",
                progress_detail="Le fichier reste disponible pour relancer.",
                eta_seconds=None,
                error="",
                finished_at=utc_now(),
            )
            return record
        except Exception as exc:
            message = str(exc)
            if not isinstance(exc, WisperAutoError):
                message = f"Erreur inattendue : {exc}"

            failed_path = ""
            if source_path.exists():
                failed_destination = safe_destination(self.config.failed_dir, source_path.name)
                try:
                    shutil.move(str(source_path), failed_destination)
                    failed_path = str(failed_destination)
                except OSError:
                    failed_path = ""

            record = self._update(
                record,
                progress_callback,
                status=STATUS_ERROR,
                progress=0,
                phase="error",
                message=message,
                progress_detail="Consultez le message d'erreur",
                eta_seconds=None,
                error=message,
                failed_path=failed_path,
                finished_at=utc_now(),
            )
            return record
        finally:
            if created_temp_wav and temp_wav_path and temp_wav_path.exists() and not self.config.keep_intermediate_wav:
                try:
                    temp_wav_path.unlink()
                except OSError:
                    pass

    def validate_audio_file(self, path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(path.suffix.lower(), SUPPORTED_EXTENSIONS)
        if not path.exists() or not path.is_file():
            raise FileNotReadyError()

        size = path.stat().st_size
        if size <= 0:
            raise EmptyAudioFileError()

        size_mb = size / (1024 * 1024)
        if size_mb > self.config.max_file_mb:
            raise AudioFileTooLargeError(size_mb, self.config.max_file_mb)

    def wait_until_file_ready(
        self,
        path: Path,
        stable_checks: int = 2,
        interval_seconds: float = 0.5,
        timeout_seconds: int = 120,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_size = -1
        stable_count = 0

        while time.monotonic() < deadline:
            if cancel_token:
                cancel_token.raise_if_cancelled()
            if not path.exists():
                raise FileNotReadyError()
            size = path.stat().st_size
            if size == last_size and size > 0:
                stable_count += 1
                if stable_count >= stable_checks:
                    return
            else:
                stable_count = 0
                last_size = size
            time.sleep(interval_seconds)

        raise FileNotReadyError()

    def _wait_until_file_ready(self, path: Path, cancel_token: CancellationToken) -> None:
        try:
            self.wait_until_file_ready(path, cancel_token=cancel_token)
        except TypeError as exc:
            if "cancel_token" not in str(exc):
                raise
            self.wait_until_file_ready(path)
            cancel_token.raise_if_cancelled()

    def needs_conversion(self, source_path: Path) -> bool:
        if resolve_backend_id(self.config) == BACKEND_WHISPER_CPP:
            return source_path.suffix.lower() not in WHISPER_CPP_DIRECT_EXTENSIONS
        return source_path.suffix.lower() not in DIRECT_TRANSCRIBE_EXTENSIONS

    def convert_to_wav(
        self,
        source_path: Path,
        record: JobRecord | str,
        duration: float | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        if shutil.which("ffmpeg") is None:
            raise FFmpegUnavailableError()

        job_id = record.id if isinstance(record, JobRecord) else record
        wav_path = self.config.outbox_dir / f"{job_id}.wav"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ]

        if not isinstance(record, JobRecord) or self.command_runner is not subprocess.run:
            try:
                self.command_runner(command, check=True, capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise FFmpegUnavailableError() from exc
            except subprocess.CalledProcessError as exc:
                details = exc.stderr or exc.stdout or str(exc)
                raise FFmpegConversionError(details) from exc
            return wav_path

        progress_command = command[:-1] + ["-progress", "pipe:1", "-nostats", str(wav_path)]
        started_at = time.monotonic()
        process = subprocess.Popen(
            progress_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        last_progress = 8
        output: list[str] = []
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
                if cancel_token:
                    cancel_token.raise_if_cancelled()

                try:
                    while True:
                        line = lines.get_nowait()
                        if line == "":
                            output_done = True
                            break
                        line = line.strip()
                        if line:
                            output.append(line)
                        if duration and line.startswith("out_time_ms="):
                            out_ms = safe_float(line.split("=", 1)[1]) or 0
                            ratio = max(0.0, min(1.0, out_ms / 1_000_000 / duration))
                            progress = max(last_progress, min(19, 8 + int(ratio * 11)))
                            if progress > last_progress:
                                last_progress = progress
                                elapsed = time.monotonic() - started_at
                                eta = estimate_eta(elapsed, ratio)
                                self._update(
                                    record,
                                    progress_callback,
                                    progress=progress,
                                    phase="converting",
                                    message="Conversion audio en cours.",
                                    progress_detail=f"{int(ratio * 100)}% conversion",
                                    eta_seconds=eta,
                                )
                except queue.Empty:
                    pass

                returncode = process.poll()
                if returncode is not None and output_done:
                    break
                time.sleep(0.1)
        except OperationCancelledError:
            terminate_process(process)
            raise

        if returncode != 0:
            details = "\n".join(output[-20:]) or "conversion impossible"
            raise FFmpegConversionError(details)

        return wav_path

    def probe_duration(self, audio_path: Path) -> float | None:
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
            str(audio_path),
        ]
        try:
            result = self.command_runner(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        return safe_float(result.stdout.strip())

    def transcribe_audio(
        self,
        audio_path: Path,
        record: JobRecord,
        allow_model_download: bool,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        cancel_token = cancel_token or CancellationToken()
        load_started_at = time.monotonic()
        engine = self._get_engine(allow_model_download)
        load_seconds = time.monotonic() - load_started_at
        cancel_token.raise_if_cancelled()

        record = self._update(
            record,
            progress_callback,
            progress=25,
            phase="transcribing",
            message="Modele pret. Transcription en cours.",
            progress_detail=f"Profil {self.config.transcription_profile} - modele charge en {load_seconds:.1f}s",
            eta_seconds=None,
        )

        stem = sanitize_stem(Path(record.source_name).stem)
        raw_path = self.config.outbox_dir / f"{record.id}_{stem}_raw.txt"
        raw_part_path = raw_path.with_suffix(raw_path.suffix + ".part")
        started_at = time.monotonic()
        last_progress = record.progress

        def update_transcription_progress(ratio: float, detail: str = "") -> None:
            nonlocal record, last_progress
            ratio = max(0.0, min(1.0, ratio))
            progress = max(25, min(92, 25 + int(ratio * 67)))
            if progress <= last_progress:
                return
            last_progress = progress
            elapsed = time.monotonic() - started_at
            eta = estimate_eta(elapsed, ratio)
            if not detail:
                detail = f"{int(ratio * 100)}% transcription"
            if eta:
                detail = f"{detail} - {format_eta(eta)}"
            record = self._update(
                record,
                progress_callback,
                progress=progress,
                phase="transcribing",
                message="Transcription en cours.",
                progress_detail=detail,
                eta_seconds=eta,
            )

        if hasattr(engine, "transcribe_with_progress"):
            segments, info = engine.transcribe_with_progress(
                audio_path,
                progress_update=update_transcription_progress,
                cancel_token=cancel_token,
            )
        else:
            segments, info = engine.transcribe(audio_path)

        duration = getattr(info, "duration", None) or record.duration_seconds
        if duration and not record.duration_seconds:
            record = self._update(record, progress_callback, duration_seconds=float(duration))

        try:
            with raw_part_path.open("w", encoding="utf-8") as handle:
                for segment in segments:
                    cancel_token.raise_if_cancelled()
                    text = getattr(segment, "text", "").strip()
                    if text:
                        handle.write(text + "\n")
                        handle.flush()
                    end = getattr(segment, "end", None)
                    if duration and end:
                        ratio = max(0.0, min(1.0, float(end) / float(duration)))
                        detail = f"{format_duration(float(end))} / {format_duration(float(duration))}"
                        update_transcription_progress(ratio, detail)
        except Exception:
            if raw_part_path.exists():
                try:
                    raw_part_path.unlink()
                except OSError:
                    pass
            raise

        cancel_token.raise_if_cancelled()
        os.replace(raw_part_path, raw_path)
        record = self._update(
            record,
            progress_callback,
            progress=max(record.progress, 94),
            phase="postprocess",
            message="Post-traitement intelligent en cours.",
            progress_detail="Generation des versions brute, nettoyee et intelligente",
            eta_seconds=None,
        )
        raw_text = raw_path.read_text(encoding="utf-8")
        post_processor = PostProcessor(self.config.commands_path)
        result = post_processor.build_outputs(raw_text)

        output_paths: dict[str, str] = {}
        for mode, content in result.outputs.items():
            output_path = self.config.outbox_dir / f"{record.id}_{stem}_{mode}.txt"
            part_path = output_path.with_suffix(output_path.suffix + ".part")
            part_path.write_text(content, encoding="utf-8")
            os.replace(part_path, output_path)
            output_paths[mode] = str(output_path)

        selected_mode = self.config.output_mode if self.config.output_mode in output_paths else MODE_SMART
        self._update(
            record,
            progress_callback,
            raw_transcript_path=str(raw_path),
            outputs=output_paths,
            transcript_path=output_paths[selected_mode],
            progress=max(record.progress, 97),
            phase="postprocess",
            message="Versions de sortie preparees.",
            progress_detail="",
        )
        return Path(output_paths[selected_mode])

    def _get_engine(self, allow_model_download: bool) -> Engine:
        backend_id = resolve_backend_id(self.config)
        local_model = self.config.local_model_path(backend_id)
        key = (
            backend_id,
            self.config.model_size,
            self.config.compute_type,
            self.config.device,
            self.config.cpu_threads,
            self.config.num_workers,
            self.config.batch_size,
            self.config.whisper_cpp_threads,
            self.config.whisper_cpp_beam_size,
            self.config.whisper_cpp_best_of,
            self.config.vad_silence_ms,
            self.config.transcription_profile,
            str(local_model) if local_model else None,
        )
        with self._engine_lock:
            if self._engine is not None and self._engine_key == key:
                return self._engine
            self._engine = self.engine_factory(self.config, allow_model_download)
            self._engine_key = key
            return self._engine

    def _update(
        self,
        record: JobRecord,
        progress_callback: ProgressCallback | None,
        **changes,
    ) -> JobRecord:
        message = changes.get("message")
        updated = self.store.update(record, **changes)
        self._notify(progress_callback, updated, str(message or updated.message or ""))
        return updated

    @staticmethod
    def _notify(
        progress_callback: ProgressCallback | None,
        record: JobRecord,
        message: str,
    ) -> None:
        if progress_callback:
            progress_callback(record, message)


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_eta(elapsed_seconds: float, ratio: float) -> float | None:
    if ratio <= 0.01:
        return None
    return max(0.0, elapsed_seconds * (1.0 - ratio) / ratio)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes:02d}:{rest:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{rest:02d}"


def terminate_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def iter_supported_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.endswith(".part")
    )
