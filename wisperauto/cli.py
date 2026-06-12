"""Command line entry points for WisperAuto."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import AppConfig
from .installers import download_model
from .jobs import JobStore
from .pipeline import TranscriptionPipeline, check_environment, iter_supported_files


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _print_preflight(config: AppConfig) -> None:
    report = check_environment(config)
    print(f"[INFO] Dossier WisperAuto : {config.home}")
    print(f"[INFO] Moteur : {report.backend_label} ({report.device}, {report.compute_type})")
    print(f"[INFO] {config.model_status_label(report.backend_id)}")
    for message in report.messages:
        print(f"[ATTENTION] {message}")


def run_watch(config: AppConfig, allow_model_download: bool = False) -> int:
    pipeline = TranscriptionPipeline(config, JobStore(config.history_path))
    seen: dict[Path, tuple[int, int]] = {}
    print("[START] Surveillance du dossier inbox")
    print(f"[INFO] Inbox : {config.inbox_dir}")
    _print_preflight(config)
    report = check_environment(config)
    if allow_model_download and not config.local_model_path(report.backend_id):
        download_model(config, backend=report.backend_id, logger=lambda message: print(f"[MODELE] {message}"))

    try:
        while True:
            for path in iter_supported_files(config.inbox_dir):
                signature = _file_signature(path)
                if signature is None:
                    continue
                if seen.get(path) == signature:
                    continue
                seen[path] = signature
                try:
                    record = pipeline.process_file(
                        path,
                        allow_model_download=allow_model_download,
                        progress_callback=lambda item, msg: print(
                            f"[{item.status_label}] {item.source_name} - {msg}"
                        ),
                    )
                finally:
                    if not path.exists():
                        seen.pop(path, None)
                if record.status == "error":
                    print(f"[ERREUR] {record.source_name} : {record.error}")
                else:
                    print(f"[OK] {record.source_name} -> {record.transcript_path}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[STOP] Surveillance arretee")
        return 0


def run_once(config: AppConfig, path: Path, allow_model_download: bool = False) -> int:
    pipeline = TranscriptionPipeline(config, JobStore(config.history_path))
    report = check_environment(config)
    if allow_model_download and not config.local_model_path(report.backend_id):
        download_model(config, backend=report.backend_id, logger=lambda message: print(f"[MODELE] {message}"))
    inbox_path = pipeline.import_audio_file(path)
    record = pipeline.process_file(
        inbox_path,
        allow_model_download=allow_model_download,
        progress_callback=lambda item, msg: print(
            f"[{item.status_label}] {item.source_name} - {msg}"
        ),
    )
    if record.status == "error":
        print(f"[ERREUR] {record.error}", file=sys.stderr)
        return 1
    print(f"[OK] Transcription : {record.transcript_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WisperAuto transcription locale")
    parser.add_argument("--watch", action="store_true", help="surveiller le dossier inbox")
    parser.add_argument("--once", type=Path, help="transcrire un fichier puis quitter")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="autoriser le telechargement initial du modele si absent localement",
    )
    args = parser.parse_args(argv)

    config = AppConfig.from_env()
    config.ensure_directories()

    if args.once:
        return run_once(config, args.once, args.allow_model_download)
    if args.watch:
        return run_watch(config, args.allow_model_download)

    from .ui import run_app

    run_app(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
