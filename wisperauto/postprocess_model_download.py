"""Subprocess entry point for visible LLM post-processing model downloads."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, DEFAULT_POSTPROCESS_MODEL_FILE, DEFAULT_POSTPROCESS_MODEL_REPO


BYTES_PER_MB = 1024 * 1024
XET_LOG_TAIL_BYTES = 192 * 1024


@dataclass(frozen=True)
class XetProgress:
    observed_bytes: int | None = None
    predicted_bandwidth: int | None = None
    current_concurrency: int | None = None
    completed_transmissions: int | None = None
    log_path: Path | None = None


def file_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_file():
        return 0.0
    try:
        return path.stat().st_size / BYTES_PER_MB
    except OSError:
        return 0.0


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total_bytes += item.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0.0
    return total_bytes / BYTES_PER_MB


def _read_tail(path: Path, max_bytes: int = XET_LOG_TAIL_BYTES) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def latest_xet_progress(logs_dir: Path | None = None, min_mtime: float | None = None) -> XetProgress | None:
    logs_dir = logs_dir or Path.home() / ".cache" / "huggingface" / "xet" / "logs"
    if not logs_dir.exists():
        return None

    candidates: list[Path] = []
    for log_path in logs_dir.glob("xet_*.log"):
        try:
            if min_mtime is not None and log_path.stat().st_mtime < min_mtime:
                continue
        except OSError:
            continue
        candidates.append(log_path)
    if not candidates:
        return None

    log_path = max(candidates, key=lambda item: item.stat().st_mtime)
    text = _read_tail(log_path)
    if not text:
        return None

    observed_bytes: int | None = None
    predicted_bandwidth: int | None = None
    current_concurrency: int | None = None
    completed_transmissions: int | None = None
    for line in text.splitlines():
        observed_match = re.search(r"observed bytes sent so far = (\d+)", line)
        if observed_match:
            observed_bytes = int(observed_match.group(1))
        bandwidth_match = re.search(r"predicted bandwidth = (\d+)", line)
        if bandwidth_match:
            predicted_bandwidth = int(bandwidth_match.group(1))
        concurrency_match = re.search(r"Current concurrency = (\d+)", line)
        if concurrency_match:
            current_concurrency = int(concurrency_match.group(1))
        transmissions_match = re.search(r"completed transmissions = (\d+)", line)
        if transmissions_match:
            completed_transmissions = int(transmissions_match.group(1))

    if (
        observed_bytes is None
        and predicted_bandwidth is None
        and current_concurrency is None
        and completed_transmissions is None
    ):
        return None
    return XetProgress(
        observed_bytes=observed_bytes,
        predicted_bandwidth=predicted_bandwidth,
        current_concurrency=current_concurrency,
        completed_transmissions=completed_transmissions,
        log_path=log_path,
    )


def _format_download_status(
    elapsed: int,
    target_path: Path,
    target_dir: Path,
    xet_progress: XetProgress | None,
    stalled_seconds: int,
) -> str:
    final_mb = file_size_mb(target_path)
    local_cache_mb = directory_size_mb(target_dir / ".cache" / "huggingface" / "download")
    target_dir_mb = directory_size_mb(target_dir)
    xet_mb = (xet_progress.observed_bytes or 0) / BYTES_PER_MB if xet_progress else 0.0

    parts = [f"Telechargement modele LLM en cours depuis {elapsed}s"]
    if final_mb > 0:
        parts.append(f"fichier final {final_mb:.1f} Mo")
    elif local_cache_mb > 0:
        parts.append(f"cache local {local_cache_mb:.1f} Mo")
    elif xet_mb > 0:
        parts.append(f"cache Xet {xet_mb:.1f} Mo recus")
    elif target_dir_mb > 0:
        parts.append(f"dossier cible {target_dir_mb:.1f} Mo")
    else:
        parts.append("connexion Hugging Face en cours, aucun octet local visible")

    if xet_progress and xet_progress.predicted_bandwidth is not None:
        parts.append(f"debit Xet estime {xet_progress.predicted_bandwidth / 1024:.0f} Ko/s")
    if xet_progress and xet_progress.current_concurrency is not None:
        parts.append(f"concurrence Xet effective {xet_progress.current_concurrency}")
    if xet_progress and xet_progress.completed_transmissions is not None:
        parts.append(f"blocs termines {xet_progress.completed_transmissions}")
    if xet_mb > 0 and final_mb == 0 and local_cache_mb == 0:
        parts.append("le fichier final peut apparaitre seulement en fin de reconstruction")
    if stalled_seconds >= 90:
        parts.append(f"aucune progression visible depuis {stalled_seconds}s")

    return " - ".join(parts) + "."


def start_heartbeat(target_path: Path, stop_event: threading.Event, interval_seconds: int = 10) -> None:
    started_at = time.monotonic()
    wall_started_at = time.time()
    target_dir = target_path.parent
    last_progress_value = 0.0
    last_progress_time = started_at

    def run() -> None:
        nonlocal last_progress_value, last_progress_time
        while not stop_event.wait(interval_seconds):
            now = time.monotonic()
            elapsed = int(now - started_at)
            xet_progress = latest_xet_progress(min_mtime=wall_started_at - 30)
            progress_value = max(
                file_size_mb(target_path),
                directory_size_mb(target_dir / ".cache" / "huggingface" / "download"),
                ((xet_progress.observed_bytes or 0) / BYTES_PER_MB if xet_progress else 0.0),
            )
            if progress_value > last_progress_value + 0.1:
                last_progress_value = progress_value
                last_progress_time = now
            stalled_seconds = int(now - last_progress_time)
            print(
                _format_download_status(elapsed, target_path, target_dir, xet_progress, stalled_seconds),
                flush=True,
            )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telecharger le modele LLM de post-traitement WisperAuto")
    parser.add_argument("--home", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_POSTPROCESS_MODEL_REPO)
    parser.add_argument("--filename", default=DEFAULT_POSTPROCESS_MODEL_FILE)
    parser.add_argument("--disable-hf-xet", action="store_true")
    parser.add_argument("--hf-fast-download", action="store_true")
    parser.add_argument("--hf-xet-concurrency", type=int, default=32)
    args = parser.parse_args(argv)

    if args.hf_fast_download:
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(max(1, args.hf_xet_concurrency))
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        print(
            "Mode Hugging Face rapide active : hf-xet haute performance, "
            f"{os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS']} connexions concurrentes.",
            flush=True,
        )
    elif args.disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("Backend Hugging Face Xet desactive pour un telechargement plus lisible.", flush=True)
    if os.environ.get("HF_TOKEN"):
        print("Token Hugging Face detecte pour ce telechargement.", flush=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERREUR: huggingface_hub est requis pour telecharger ce modele.", flush=True)
        return 1

    config = AppConfig(
        home=Path(args.home),
        postprocess_model_repo=args.repo_id,
        postprocess_model_file=args.filename,
    )
    config.ensure_directories()
    target_path = config.postprocess_models_dir / args.filename
    print(f"Depot Hugging Face : {args.repo_id}", flush=True)
    print(f"Fichier : {args.filename}", flush=True)
    print(f"Dossier cible : {config.postprocess_models_dir}", flush=True)
    if args.hf_fast_download:
        print(
            "Note : avec hf-xet, le fichier final peut rester a 0 Mo pendant que le cache Xet se remplit.",
            flush=True,
        )

    stop_event = threading.Event()
    start_heartbeat(target_path, stop_event)
    try:
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(config.postprocess_models_dir),
            local_files_only=False,
        )
    except Exception as exc:
        stop_event.set()
        print(f"ERREUR: {exc}", flush=True)
        return 1

    stop_event.set()
    downloaded_path = Path(downloaded)
    if downloaded_path != target_path and downloaded_path.exists() and not target_path.exists():
        try:
            shutil.copy2(downloaded_path, target_path)
        except OSError:
            pass
    if not target_path.exists():
        print(f"ERREUR: fichier telecharge introuvable : {target_path}", flush=True)
        return 1
    print(f"Modele LLM pret : {target_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
