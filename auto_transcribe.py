import time
import shutil
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from faster_whisper import WhisperModel

BASE_DIR = Path("C:/WhisperAuto")

INBOX = BASE_DIR / "inbox"
OUTBOX = BASE_DIR / "outbox"
PROCESSED = BASE_DIR / "processed"

MODEL_SIZE = "large-v3"
LANGUAGE = "fr"

model = WhisperModel(
    MODEL_SIZE,
    compute_type="int8"
)

for folder in [INBOX, OUTBOX, PROCESSED]:
    folder.mkdir(parents=True, exist_ok=True)

def wait_until_file_ready(path: Path):
    last_size = -1

    while True:
        size = path.stat().st_size

        if size == last_size:
            return

        last_size = size
        time.sleep(2)

def convert_to_wav(input_path: Path) -> Path:
    wav_path = OUTBOX / f"{input_path.stem}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(wav_path)
    ]

    subprocess.run(command, check=True)

    return wav_path

def transcribe_audio(wav_path: Path):
    txt_path = OUTBOX / f"{wav_path.stem}.txt"

    segments, info = model.transcribe(
        str(wav_path),
        language=LANGUAGE,
        vad_filter=True
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")

    return txt_path

def process_file(path: Path):
    try:
        print(f"[INFO] Traitement : {path.name}")

        wait_until_file_ready(path)

        wav_path = convert_to_wav(path)

        txt_path = transcribe_audio(wav_path)

        shutil.move(
            str(path),
            PROCESSED / path.name
        )

        print(f"[OK] Transcription : {txt_path.name}")

    except Exception as e:
        print(f"[ERREUR] {path.name} : {e}")

class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):

        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() in [
            ".ds2",
            ".dss",
            ".mp3",
            ".wav",
            ".m4a"
        ]:
            process_file(path)

if __name__ == "__main__":

    print("[START] Surveillance du dossier inbox")

    observer = Observer()

    observer.schedule(
        AudioHandler(),
        str(INBOX),
        recursive=False
    )

    observer.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()

    observer.join()