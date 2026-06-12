"""Local job history stored as append-only JSON lines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Iterable


STATUS_READY = "ready"
STATUS_QUEUED = "queued"
STATUS_CONVERTING = "converting"
STATUS_TRANSCRIBING = "transcribing"
STATUS_POSTPROCESSING = "postprocessing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_IN_PROGRESS = {
    STATUS_QUEUED,
    STATUS_CONVERTING,
    STATUS_TRANSCRIBING,
    STATUS_POSTPROCESSING,
}

STATUS_LABELS = {
    STATUS_READY: "Pret",
    STATUS_QUEUED: "En attente",
    STATUS_CONVERTING: "Conversion",
    STATUS_TRANSCRIBING: "Transcription",
    STATUS_POSTPROCESSING: "Post-traitement",
    STATUS_DONE: "Termine",
    STATUS_ERROR: "Erreur",
    STATUS_CANCELLED: "Annule",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class JobRecord:
    id: str
    source_name: str
    source_path: str
    status: str = STATUS_QUEUED
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    transcript_path: str = ""
    raw_transcript_path: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    processed_path: str = ""
    failed_path: str = ""
    error: str = ""
    progress: int = 0
    phase: str = ""
    message: str = ""
    progress_detail: str = ""
    eta_seconds: float | None = None
    duration_seconds: float | None = None
    started_at: str = ""
    finished_at: str = ""
    model_size: str = ""
    backend: str = ""
    transcription_profile: str = ""

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "JobRecord":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in payload.items() if key in valid_keys}
        return cls(**filtered)


class JobStore:
    def __init__(self, history_path: Path):
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: JobRecord) -> None:
        with self._lock:
            record.updated_at = utc_now()
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def update(self, record: JobRecord, **changes) -> JobRecord:
        for key, value in changes.items():
            setattr(record, key, value)
        self.append(record)
        return record

    def latest(self) -> list[JobRecord]:
        with self._lock:
            return self._latest_unlocked()

    def delete(self, job_id: str) -> JobRecord | None:
        with self._lock:
            records = self._latest_unlocked()
            deleted = next((record for record in records if record.id == job_id), None)
            kept = [record for record in records if record.id != job_id]
            self._rewrite_unlocked(kept)
            return deleted

    def replace_all(self, records: Iterable[JobRecord]) -> None:
        with self._lock:
            self._rewrite_unlocked(list(records))

    def iter_records(self) -> Iterable[JobRecord]:
        with self._lock:
            return self._iter_records_unlocked()

    def _latest_unlocked(self) -> list[JobRecord]:
        records: dict[str, JobRecord] = {}
        for record in self._iter_records_unlocked():
            records[record.id] = record
        return sorted(records.values(), key=lambda item: item.created_at, reverse=True)

    def _iter_records_unlocked(self) -> list[JobRecord]:
        if not self.history_path.exists():
            return []

        parsed: list[JobRecord] = []
        with self.history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed.append(JobRecord.from_dict(json.loads(line)))
                except (TypeError, json.JSONDecodeError):
                    continue
        return parsed

    def _rewrite_unlocked(self, records: list[JobRecord]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda item: item.created_at):
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        tmp_path.replace(self.history_path)


def recover_interrupted_records(store: JobStore, records: Iterable[JobRecord]) -> list[JobRecord]:
    """Mark jobs left in an in-progress state by a previous app session as retryable."""

    recovered: list[JobRecord] = []
    for record in records:
        if record.status not in STATUS_IN_PROGRESS:
            recovered.append(record)
            continue
        recovered.append(
            store.update(
                record,
                status=STATUS_CANCELLED,
                phase="cancelled",
                message="Traitement interrompu lors d'une session precedente.",
                progress_detail="Relancez la transcription ou supprimez cette entree.",
                eta_seconds=None,
                finished_at=utc_now(),
            )
        )
    return recovered
