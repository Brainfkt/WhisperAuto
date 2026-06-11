"""Subprocess entry point for visible model downloads."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path

from .config import BACKEND_FASTER_WHISPER, AppConfig
from .models import ensure_backend_model_available


EXPECTED_MODEL_MB = {
    "small": 466,
    "medium": 1500,
    "large-v3-turbo": 1600,
    "large-v3": 3100,
}


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def latest_xet_progress() -> str:
    logs_dir = Path.home() / ".cache" / "huggingface" / "xet" / "logs"
    if not logs_dir.exists():
        return ""

    logs = sorted(logs_dir.glob("xet_*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not logs:
        return ""

    latest = logs[0]
    try:
        lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
    except OSError:
        return ""

    observed_bytes = None
    predicted_bandwidth = None
    for line in lines:
        observed_match = re.search(r"observed bytes sent so far = (\d+)", line)
        if observed_match:
            observed_bytes = int(observed_match.group(1))
        bandwidth_match = re.search(r"predicted bandwidth = (\d+)", line)
        if bandwidth_match:
            predicted_bandwidth = int(bandwidth_match.group(1))

    if observed_bytes is None:
        return ""

    parts = [f"cache Hugging Face/Xet : {observed_bytes / (1024 * 1024):.1f} Mo recus"]
    if predicted_bandwidth:
        parts.append(f"debit estime {predicted_bandwidth / 1024:.0f} Ko/s")
    return ", ".join(parts)


def file_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
    return count


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"ETA ~{seconds}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"ETA ~{minutes}min {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"ETA ~{hours}h {minutes:02d}min"


def start_heartbeat(
    target_dir: Path,
    stop_event: threading.Event,
    model_size: str,
    backend: str,
    interval_seconds: int = 10,
) -> threading.Thread:
    started_at = time.monotonic()
    expected_mb = EXPECTED_MODEL_MB.get(model_size)
    last_size = 0.0
    last_time = started_at

    def run() -> None:
        nonlocal last_size, last_time
        while not stop_event.wait(interval_seconds):
            elapsed = int(time.monotonic() - started_at)
            now = time.monotonic()
            size_mb = directory_size_mb(target_dir)
            delta_mb = max(0.0, size_mb - last_size)
            delta_seconds = max(0.1, now - last_time)
            rate_mb_s = delta_mb / delta_seconds
            last_size = size_mb
            last_time = now
            eta = None
            if expected_mb and rate_mb_s > 0:
                eta = max(0.0, (expected_mb - size_mb) / rate_mb_s)
            detail = [
                f"Telechargement {backend} en cours depuis {elapsed}s",
                f"{file_count(target_dir)} fichiers visibles",
                f"{size_mb:.1f} Mo",
            ]
            if rate_mb_s > 0:
                detail.append(f"{rate_mb_s:.2f} Mo/s")
            if expected_mb:
                detail.append(f"taille attendue ~{expected_mb} Mo")
            if eta is not None:
                detail.append(format_eta(eta))
            xet = latest_xet_progress()
            if xet:
                detail.append(xet)
            print(
                " - ".join(detail) + ".",
                flush=True,
            )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telecharger le modele WisperAuto")
    parser.add_argument("--home", required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--backend", default=BACKEND_FASTER_WHISPER)
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

    config = AppConfig(
        home=Path(args.home),
        model_size=args.model_size,
        language=args.language,
        compute_type=args.compute_type,
        backend=args.backend,
        allow_model_download=True,
    )
    config.ensure_directories()

    print(f"Modele demande : {config.model_size}", flush=True)
    print(f"Moteur : {args.backend}", flush=True)
    print(f"Dossier de destination : {config.backend_model_dir(args.backend)}", flush=True)
    print(
        "Le fichier model.bin peut etre volumineux. "
        "Si le debit est faible, utilisez un modele plus petit ou un token Hugging Face.",
        flush=True,
    )
    print("Connexion au depot de modeles...", flush=True)

    stop_event = threading.Event()
    start_heartbeat(config.backend_model_dir(args.backend), stop_event, config.model_size, args.backend)
    try:
        model_path = ensure_backend_model_available(
            config,
            args.backend,
            logger=lambda message: print(message, flush=True),
        )
    except Exception as exc:
        stop_event.set()
        print(f"ERREUR: {exc}", flush=True)
        return 1

    stop_event.set()
    print(f"Modele pret : {model_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
