"""Local transcription model management."""

from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path
from typing import Callable

from .config import (
    BACKEND_FASTER_WHISPER,
    BACKEND_MLX_WHISPER,
    BACKEND_WHISPER_CPP,
    AppConfig,
    looks_like_backend_model,
    looks_like_model_dir,
)
from .errors import DependencyUnavailableError


ModelLogger = Callable[[str], None]


def resolve_model_dir(path: Path, config: AppConfig) -> Path | None:
    if looks_like_model_dir(path):
        return path
    if path.exists() and path.is_dir():
        matches = sorted(path.rglob("model.bin"))
        for match in matches:
            parent = match.parent
            if config.model_size.lower() in str(parent).lower() or path == config.model_dir:
                return parent
        if matches:
            return matches[0].parent
    return None


def ensure_model_available(config: AppConfig, logger: ModelLogger | None = None) -> Path:
    """Download the configured model into the local WisperAuto model store if needed."""

    return ensure_backend_model_available(config, BACKEND_FASTER_WHISPER, logger)


def ensure_backend_model_available(
    config: AppConfig,
    backend: str = BACKEND_FASTER_WHISPER,
    logger: ModelLogger | None = None,
) -> Path:
    if backend == BACKEND_MLX_WHISPER:
        return ensure_mlx_model_available(config, logger)
    if backend == BACKEND_WHISPER_CPP:
        return ensure_whisper_cpp_model_available(config, logger)
    return ensure_faster_whisper_model_available(config, logger)


def ensure_faster_whisper_model_available(config: AppConfig, logger: ModelLogger | None = None) -> Path:
    """Download the configured CTranslate2 model into the local WisperAuto model store."""

    logger = logger or (lambda _message: None)
    local_model = config.local_model_path(BACKEND_FASTER_WHISPER)
    if local_model:
        logger(f"Modele deja disponible : {local_model}")
        return local_model

    if importlib.util.find_spec("faster_whisper") is None:
        raise DependencyUnavailableError("faster-whisper")

    target_dir = config.backend_model_dir(BACKEND_FASTER_WHISPER)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger(f"Telechargement du modele {config.model_size}.")
    logger(f"Dossier cible : {target_dir}")

    try:
        from faster_whisper.utils import download_model

        downloaded = download_model(
            config.model_size,
            output_dir=str(target_dir),
            local_files_only=False,
        )
        if downloaded:
            downloaded_path = Path(downloaded)
            resolved = resolve_model_dir(downloaded_path, config)
            if resolved:
                config.remember_model_path(resolved, BACKEND_FASTER_WHISPER)
                logger(f"Modele pret : {resolved}")
                return resolved
    except TypeError:
        logger("API download_model differente ; utilisation du chargement WhisperModel.")
    except ImportError:
        logger("download_model indisponible ; utilisation du chargement WhisperModel.")

    from faster_whisper import WhisperModel

    WhisperModel(
        config.model_size,
        compute_type=config.compute_type,
        download_root=str(target_dir),
    )

    resolved = resolve_model_dir(target_dir, config)
    if resolved:
        config.remember_model_path(resolved, BACKEND_FASTER_WHISPER)
        logger(f"Modele pret : {resolved}")
        return resolved

    local_model = config.local_model_path(BACKEND_FASTER_WHISPER)
    if local_model:
        logger(f"Modele pret : {local_model}")
        return local_model

    raise RuntimeError(
        "Le modele semble avoir ete charge, mais aucun dossier local exploitable n'a ete trouve."
    )


MLX_MODEL_REPOS = {
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def ensure_mlx_model_available(config: AppConfig, logger: ModelLogger | None = None) -> Path:
    logger = logger or (lambda _message: None)
    local_model = config.local_model_path(BACKEND_MLX_WHISPER)
    if local_model:
        logger(f"Modele MLX deja disponible : {local_model}")
        return local_model

    if importlib.util.find_spec("mlx_whisper") is None:
        raise DependencyUnavailableError("mlx-whisper")
    if importlib.util.find_spec("huggingface_hub") is None:
        raise DependencyUnavailableError("huggingface_hub")

    from huggingface_hub import snapshot_download

    repo_id = MLX_MODEL_REPOS.get(config.model_size)
    if not repo_id:
        raise RuntimeError(f"Aucun modele MLX configure pour {config.model_size}.")

    target_dir = config.backend_model_dir(BACKEND_MLX_WHISPER)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger(f"Telechargement du modele MLX {config.model_size}.")
    logger(f"Depot Hugging Face : {repo_id}")
    logger(f"Dossier cible : {target_dir}")
    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_files_only=False,
        )
    except Exception as exc:
        if exc.__class__.__name__ == "RepositoryNotFoundError":
            raise RuntimeError(
                "Depot Hugging Face MLX introuvable ou inaccessible : "
                f"{repo_id}. Verifiez le modele choisi, le token Hugging Face, "
                "ou choisissez large-v3-turbo pour le backend MLX."
            ) from exc
        raise
    resolved = Path(downloaded)
    if looks_like_backend_model(resolved, BACKEND_MLX_WHISPER):
        config.remember_model_path(resolved, BACKEND_MLX_WHISPER)
        logger(f"Modele MLX pret : {resolved}")
        return resolved
    if looks_like_backend_model(target_dir, BACKEND_MLX_WHISPER):
        config.remember_model_path(target_dir, BACKEND_MLX_WHISPER)
        logger(f"Modele MLX pret : {target_dir}")
        return target_dir
    raise RuntimeError("Telechargement MLX termine, mais le modele reste introuvable.")


WHISPER_CPP_MODEL_URLS = {
    "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    "large-v3-turbo": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    "large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
}


def ensure_whisper_cpp_model_available(config: AppConfig, logger: ModelLogger | None = None) -> Path:
    logger = logger or (lambda _message: None)
    local_model = config.local_model_path(BACKEND_WHISPER_CPP)
    if local_model:
        logger(f"Modele whisper.cpp deja disponible : {local_model}")
        return local_model

    url = WHISPER_CPP_MODEL_URLS.get(config.model_size)
    if not url:
        raise RuntimeError(f"Aucun modele whisper.cpp configure pour {config.model_size}.")

    target_dir = config.backend_model_dir(BACKEND_WHISPER_CPP)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(url).name
    part_path = target_path.with_suffix(target_path.suffix + ".part")
    logger(f"Telechargement du modele whisper.cpp {config.model_size}.")
    logger(f"Dossier cible : {target_dir}")

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if not total_size:
            return
        downloaded_mb = block_count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        ratio = min(100, int((block_count * block_size / total_size) * 100))
        if block_count % 64 == 0:
            logger(f"Telechargement whisper.cpp : {ratio}% - {downloaded_mb:.1f}/{total_mb:.1f} Mo")

    urllib.request.urlretrieve(url, part_path, reporthook=report)
    part_path.replace(target_path)
    config.remember_model_path(target_path, BACKEND_WHISPER_CPP)
    logger(f"Modele whisper.cpp pret : {target_path}")
    return target_path
