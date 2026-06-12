"""Dependency installation helpers used by the desktop UI."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import queue
import threading
import platform
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cancel import CancellationToken
from .config import (
    BACKEND_FASTER_WHISPER,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
    AppConfig,
)
from .errors import OperationCancelledError


InstallLogger = Callable[[str], None]


class InstallUnavailableError(Exception):
    """Raised when no safe automatic installer is available."""


@dataclass(frozen=True)
class InstallPlan:
    name: str
    commands: list[list[str]]
    note: str = ""


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def faster_whisper_plan(project_root: Path) -> InstallPlan:
    requirements_path = project_root / "requirements.txt"
    if requirements_path.exists():
        install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    else:
        install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "faster-whisper>=1.0,<2.0",
        ]

    return InstallPlan(
        name="faster-whisper",
        commands=[
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            install_command,
        ],
        note="Installe la dependance Python dans l'environnement qui lance WisperAuto.",
    )


def hf_acceleration_plan() -> InstallPlan:
    return InstallPlan(
        name="Accelerateur Hugging Face",
        commands=[
            [sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub", "hf-xet"],
        ],
        note="Installe hf-xet pour les telechargements Hugging Face rapides.",
    )


def llama_cpp_python_plan() -> InstallPlan:
    return InstallPlan(
        name="Post-traitement LLM local",
        commands=[
            [sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python>=0.3,<1.0"],
        ],
        note=(
            "Installe llama-cpp-python pour generer la transcription intelligente en local. "
            "Sur Mac Apple Silicon, utilisez un Python arm64 pour de bonnes performances."
        ),
    )


def mlx_whisper_plan() -> InstallPlan:
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise InstallUnavailableError("mlx-whisper est disponible uniquement sur Mac Apple Silicon.")

    return InstallPlan(
        name="mlx-whisper",
        commands=[
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            [sys.executable, "-m", "pip", "install", "mlx-whisper>=0.4"],
        ],
        note="Installe MLX Whisper dans l'environnement Python qui lance WisperAuto.",
    )


def whisper_cpp_plan() -> InstallPlan:
    if sys.platform == "darwin" and command_exists("brew"):
        return InstallPlan(
            name="whisper.cpp",
            commands=[["brew", "install", "whisper-cpp"]],
            note="Installe le binaire whisper.cpp via Homebrew.",
        )

    raise InstallUnavailableError(
        "Aucun installateur automatique fiable de whisper.cpp n'a ete trouve. "
        "Installez le binaire whisper-cli manuellement, puis indiquez son chemin avec "
        "WISPERAUTO_WHISPER_CPP_BINARY."
    )


def backend_install_plan(backend: str, project_root: Path) -> InstallPlan:
    if backend == BACKEND_MLX_WHISPER:
        return mlx_whisper_plan()
    if backend == BACKEND_WHISPER_CPP:
        return whisper_cpp_plan()
    return faster_whisper_plan(project_root)


def ffmpeg_plan() -> InstallPlan:
    if sys.platform.startswith("win"):
        if command_exists("winget"):
            return InstallPlan(
                name="FFmpeg",
                commands=[
                    [
                        "winget",
                        "install",
                        "--id",
                        "Gyan.FFmpeg",
                        "--source",
                        "winget",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                    ]
                ],
                note="Installe FFmpeg via winget. Relancez WisperAuto si le PATH change.",
            )
        if command_exists("choco"):
            return InstallPlan(
                name="FFmpeg",
                commands=[["choco", "install", "ffmpeg", "-y"]],
                note="Installe FFmpeg via Chocolatey.",
            )
        if command_exists("scoop"):
            return InstallPlan(
                name="FFmpeg",
                commands=[["scoop", "install", "ffmpeg"]],
                note="Installe FFmpeg via Scoop.",
            )
        raise InstallUnavailableError(
            "Aucun installateur automatique trouve. Installez winget, Chocolatey ou Scoop, "
            "ou installez FFmpeg manuellement depuis https://www.gyan.dev/ffmpeg/builds/."
        )

    if sys.platform == "darwin":
        if command_exists("brew"):
            return InstallPlan(
                name="FFmpeg",
                commands=[["brew", "install", "ffmpeg"]],
                note="Installe FFmpeg via Homebrew.",
            )
        raise InstallUnavailableError(
            "Homebrew est introuvable. Installez Homebrew ou installez FFmpeg manuellement."
        )

    if command_exists("apt-get"):
        return InstallPlan(
            name="FFmpeg",
            commands=[["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "ffmpeg"]],
            note="Installe FFmpeg via apt-get. Un mot de passe administrateur peut etre demande.",
        )
    if command_exists("dnf"):
        return InstallPlan(
            name="FFmpeg",
            commands=[["sudo", "dnf", "install", "-y", "ffmpeg"]],
            note="Installe FFmpeg via dnf.",
        )
    if command_exists("pacman"):
        return InstallPlan(
            name="FFmpeg",
            commands=[["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"]],
            note="Installe FFmpeg via pacman.",
        )

    raise InstallUnavailableError(
        "Aucun gestionnaire de paquets compatible n'a ete trouve pour installer FFmpeg."
    )


def run_streamed_command(
    command: list[str],
    logger: InstallLogger,
    timeout_seconds: int | None = None,
    cancel_token: CancellationToken | None = None,
    runner=None,
    env: dict[str, str] | None = None,
) -> None:
    logger("$ " + " ".join(command))
    if cancel_token:
        cancel_token.raise_if_cancelled()
    if runner is not None:
        process = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.stdout:
            logger(process.stdout.strip())
        if process.stderr:
            logger(process.stderr.strip())
        returncode = process.returncode
    else:
        started_at = time.monotonic()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("Impossible de lire la sortie de la commande.")

        lines: queue.Queue[str] = queue.Queue()

        def read_output() -> None:
            try:
                for line in stdout:
                    lines.put(line)
            finally:
                lines.put("")

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        output_done = False
        while True:
            if cancel_token and cancel_token.cancelled:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise OperationCancelledError()
            if timeout_seconds and time.monotonic() - started_at > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(
                    f"Delai depasse apres {timeout_seconds // 60} min : {' '.join(command)}"
                )

            try:
                while True:
                    line = lines.get_nowait()
                    if line == "":
                        output_done = True
                        break
                    line = line.strip()
                    if line:
                        logger(line)
            except queue.Empty:
                pass

            returncode = process.poll()
            if returncode is not None and output_done:
                break
            time.sleep(0.2)
        returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            f"La commande a echoue avec le code {returncode}: {' '.join(command)}"
        )


def run_install_plan(
    plan: InstallPlan,
    logger: InstallLogger | None = None,
    cancel_token: CancellationToken | None = None,
    runner=None,
) -> None:
    logger = logger or (lambda _message: None)
    logger(f"Installation de {plan.name}.")
    if plan.note:
        logger(plan.note)

    for command in plan.commands:
        kwargs = {"logger": logger, "runner": runner}
        if cancel_token is not None:
            kwargs["cancel_token"] = cancel_token
        run_streamed_command(command, **kwargs)

    logger(f"Installation de {plan.name} terminee.")


def download_model(
    config: AppConfig,
    backend: str = BACKEND_FASTER_WHISPER,
    logger: InstallLogger | None = None,
    cancel_token: CancellationToken | None = None,
) -> Path:
    logger = logger or (lambda _message: None)
    local_model = config.local_model_path(backend)
    if local_model:
        logger(f"Modele deja disponible : {local_model}")
        return local_model

    if cancel_token:
        cancel_token.raise_if_cancelled()
    logger("Preparation du telechargement du modele.")
    env = os.environ.copy()
    if config.hf_token.strip():
        env["HF_TOKEN"] = config.hf_token.strip()
        logger("Token Hugging Face detecte : il sera transmis au telechargement.")
    if config.hf_fast_download:
        env.pop("HF_HUB_DISABLE_XET", None)
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(max(1, config.hf_xet_concurrency))
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        logger(
            "Mode Hugging Face rapide active : Xet haute performance, "
            f"{env['HF_XET_NUM_CONCURRENT_RANGE_GETS']} connexions concurrentes."
        )
    command = [
        sys.executable,
        "-m",
        "wisperauto.model_download",
        "--home",
        str(config.home),
        "--model-size",
        config.model_size,
        "--language",
        config.language,
        "--compute-type",
        config.compute_type,
        "--backend",
        backend,
    ]
    if config.hf_fast_download:
        command.append("--hf-fast-download")
        command.extend(["--hf-xet-concurrency", str(max(1, config.hf_xet_concurrency))])
    elif config.disable_hf_xet:
        command.append("--disable-hf-xet")
    kwargs = {
        "logger": logger,
        "timeout_seconds": config.model_download_timeout_minutes * 60,
        "env": env,
    }
    if cancel_token is not None:
        kwargs["cancel_token"] = cancel_token
    run_streamed_command(command, **kwargs)
    model_path = config.local_model_path(backend)
    if not model_path:
        raise RuntimeError("Telechargement termine, mais le modele local reste introuvable.")
    logger(f"Modele disponible : {model_path}")
    return model_path


def download_postprocess_model(
    config: AppConfig,
    logger: InstallLogger | None = None,
    cancel_token: CancellationToken | None = None,
) -> Path:
    logger = logger or (lambda _message: None)
    local_model = config.local_postprocess_model_path()
    if local_model:
        logger(f"Modele de post-traitement deja disponible : {local_model}")
        return local_model

    if cancel_token:
        cancel_token.raise_if_cancelled()
    config.ensure_directories()
    env = os.environ.copy()
    if config.hf_token.strip():
        env["HF_TOKEN"] = config.hf_token.strip()
        logger("Token Hugging Face detecte : il sera transmis au telechargement.")
    if config.hf_fast_download:
        env.pop("HF_HUB_DISABLE_XET", None)
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(max(1, config.hf_xet_concurrency))
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        logger(
            "Mode Hugging Face rapide active : Xet haute performance, "
            f"{env['HF_XET_NUM_CONCURRENT_RANGE_GETS']} connexions concurrentes."
        )

    command = [
        sys.executable,
        "-m",
        "wisperauto.postprocess_model_download",
        "--home",
        str(config.home),
        "--repo-id",
        config.postprocess_model_repo,
        "--filename",
        config.postprocess_model_file,
    ]
    if config.hf_fast_download:
        command.append("--hf-fast-download")
        command.extend(["--hf-xet-concurrency", str(max(1, config.hf_xet_concurrency))])
    elif config.disable_hf_xet:
        command.append("--disable-hf-xet")

    kwargs = {
        "logger": logger,
        "timeout_seconds": config.model_download_timeout_minutes * 60,
        "env": env,
    }
    if cancel_token is not None:
        kwargs["cancel_token"] = cancel_token
    run_streamed_command(command, **kwargs)

    model_path = config.local_postprocess_model_path()
    if not model_path:
        raise RuntimeError("Telechargement termine, mais le modele LLM local reste introuvable.")
    logger(f"Modele de post-traitement disponible : {model_path}")
    return model_path
