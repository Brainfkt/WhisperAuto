"""Domain errors surfaced as clear user-facing messages."""


class WisperAutoError(Exception):
    """Base class for predictable WisperAuto failures."""

    user_message = "Une erreur inattendue est survenue."

    def __str__(self):
        return self.user_message


class OperationCancelledError(WisperAutoError):
    user_message = "Operation annulee par l'utilisateur."


class UnsupportedFormatError(WisperAutoError):
    def __init__(self, suffix, supported):
        self.suffix = suffix or "(sans extension)"
        self.supported = supported
        self.user_message = (
            f"Format audio non supporte : {self.suffix}. "
            f"Formats acceptes : {', '.join(sorted(supported))}."
        )


class EmptyAudioFileError(WisperAutoError):
    user_message = "Le fichier audio est vide."


class AudioFileTooLargeError(WisperAutoError):
    def __init__(self, size_mb, max_mb):
        self.user_message = (
            f"Fichier trop volumineux ({size_mb:.1f} Mo). "
            f"Limite configuree : {max_mb} Mo."
        )


class AudioTooLongError(WisperAutoError):
    def __init__(self, duration_min, max_min):
        self.user_message = (
            f"Audio trop long ({duration_min:.1f} min). "
            f"Limite configuree : {max_min} min."
        )


class FileNotReadyError(WisperAutoError):
    user_message = "Le fichier n'est pas encore stable ou a ete supprime pendant la copie."


class FFmpegUnavailableError(WisperAutoError):
    user_message = (
        "FFmpeg est introuvable. Installez FFmpeg puis ajoutez son dossier bin au PATH."
    )


class FFmpegConversionError(WisperAutoError):
    def __init__(self, details):
        details = details.strip() if details else "conversion impossible"
        self.user_message = f"Conversion audio impossible : {details}"


class ModelUnavailableError(WisperAutoError):
    user_message = (
        "Modele Whisper absent en local. Autorisez le telechargement initial ou "
        "definissez WISPERAUTO_MODEL_PATH vers un modele deja present."
    )


class DependencyUnavailableError(WisperAutoError):
    def __init__(self, dependency):
        self.user_message = (
            f"Dependance Python manquante : {dependency}. "
            "Installez les dependances avec pip install -r requirements.txt."
        )


class PostProcessUnavailableError(WisperAutoError):
    def __init__(self, details):
        details = str(details).strip() if details else "modele LLM local indisponible"
        self.user_message = (
            "Post-traitement intelligent indisponible : "
            f"{details}. La transcription brute est conservee."
        )
