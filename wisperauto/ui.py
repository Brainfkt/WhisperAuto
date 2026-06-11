"""Tkinter desktop interface for WisperAuto."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from tkinter import END, LEFT, VERTICAL, W, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from .benchmark import benchmark_backends, recommendation_text
from .cancel import CancellationToken
from .backends import BACKEND_LABELS, BACKEND_ORDER, backend_health, resolve_backend_id
from .config import (
    APP_NAME,
    BACKEND_AUTO,
    PROFILE_BALANCED,
    PROFILE_FAST,
    PROFILE_PRECISE,
    SUPPORTED_EXTENSIONS,
    AppConfig,
)
from .errors import OperationCancelledError
from .installers import (
    InstallPlan,
    InstallUnavailableError,
    backend_install_plan,
    download_model,
    ffmpeg_plan,
    run_install_plan,
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
)
from .pipeline import TranscriptionPipeline, check_environment, format_eta
from .postprocess import MODE_LABELS, MODE_RAW, MODE_CLEANED, MODE_SMART, MODE_REPORT


MODE_ORDER = (MODE_RAW, MODE_CLEANED, MODE_SMART, MODE_REPORT)
MODE_BY_LABEL = {label: mode for mode, label in MODE_LABELS.items()}
MODEL_CHOICES = ("small", "medium", "large-v3-turbo", "large-v3")
PROFILE_LABELS = {
    PROFILE_FAST: "Rapide",
    PROFILE_BALANCED: "Equilibre",
    PROFILE_PRECISE: "Precis",
}
PROFILE_BY_LABEL = {label: profile for profile, label in PROFILE_LABELS.items()}
BACKEND_BY_LABEL = {label: backend for backend, label in BACKEND_LABELS.items()}
RETRYABLE_STATUSES = {STATUS_READY, STATUS_CANCELLED, STATUS_ERROR}
RUNNING_STATUSES = {STATUS_QUEUED, STATUS_CONVERTING, STATUS_TRANSCRIBING}


class WisperAutoApp:
    def __init__(self, root: tk.Tk, config: AppConfig):
        self.root = root
        self.config = config
        self.store = JobStore(config.history_path)
        self.pipeline = TranscriptionPipeline(config, self.store)
        self.worker_queue: queue.Queue[JobRecord | None] = queue.Queue()
        self.ui_queue: queue.Queue[tuple[JobRecord, str]] = queue.Queue()
        self.jobs: dict[str, JobRecord] = {}
        self.tree_items: dict[str, str] = {}
        self.queued_ids: set[str] = set()
        self.active_job_id: str | None = None
        self.current_cancel_token: CancellationToken | None = None
        self.allow_model_download = config.allow_model_download
        self.installing_dependency = False
        self.settings_dialog: SettingsDialog | None = None

        default_mode = config.output_mode if config.output_mode in MODE_LABELS else MODE_SMART
        self.output_mode = tk.StringVar(value=MODE_LABELS[default_mode])

        self._configure_window()
        self._build_styles()
        self._build_layout()
        self._load_history()
        self._show_preflight()
        self._start_worker()
        self.root.after(200, self._drain_ui_queue)

    def _configure_window(self) -> None:
        self.root.title(APP_NAME)
        self.root.geometry("1120x700")
        self.root.minsize(940, 580)
        self.root.configure(bg="#f3f4f6")

    def _build_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#f3f4f6")
        style.configure("Surface.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#ffffff", foreground="#111827")
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"), foreground="#111827")
        style.configure("Muted.TLabel", foreground="#6b7280", background="#ffffff")
        style.configure("Status.TLabel", foreground="#374151", background="#ffffff")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("Danger.TButton", padding=(12, 7))
        style.configure("TButton", padding=(10, 6))
        style.configure("Treeview", rowheight=34, font=("Segoe UI", 10), background="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TProgressbar", thickness=10)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Surface.TFrame", padding=(16, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(3, weight=1)

        ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=0, padx=(0, 18))
        self.privacy_label = ttk.Label(
            header,
            text="Local",
            style="Status.TLabel",
            foreground="#15803d",
        )
        self.privacy_label.grid(row=0, column=1, padx=(0, 18))
        self.model_label = ttk.Label(header, text=self.config.model_status_label(), style="Muted.TLabel")
        self.model_label.grid(row=0, column=2, sticky="w")

        actions = ttk.Frame(header, style="Surface.TFrame")
        actions.grid(row=0, column=4, sticky="e")
        ttk.Button(actions, text="Ajouter fichiers", style="Primary.TButton", command=self._import_audio).pack(
            side=LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Parametres", command=self._show_settings).pack(side=LEFT)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        left = ttk.Frame(body, style="Surface.TFrame", padding=(12, 12))
        right = ttk.Frame(body, style="Surface.TFrame", padding=(14, 12))
        body.add(left, weight=2)
        body.add(right, weight=3)

        self._build_queue_panel(left)
        self._build_transcript_panel(right)
        self._build_footer()

    def _build_queue_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        title_row = ttk.Frame(parent, style="Surface.TFrame")
        title_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        title_row.columnconfigure(0, weight=1)
        ttk.Label(title_row, text="File et historique", style="Header.TLabel", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.queue_count_label = ttk.Label(title_row, text="0 element", style="Muted.TLabel")
        self.queue_count_label.grid(row=0, column=1, sticky="e")

        buttons = ttk.Frame(parent, style="Surface.TFrame")
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        self.transcribe_selected_button = ttk.Button(
            buttons,
            text="Transcrire selection",
            style="Primary.TButton",
            command=self._transcribe_selected,
        )
        self.transcribe_selected_button.pack(side=LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Tout transcrire", command=self._transcribe_all).pack(side=LEFT, padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Annuler", command=self._cancel_selected_or_active)
        self.cancel_button.pack(side=LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Supprimer", command=self._delete_selected).pack(side=LEFT)

        self.history = ttk.Treeview(
            parent,
            columns=("status", "progress", "date"),
            show="tree headings",
            selectmode="extended",
        )
        self.history.heading("#0", text="Fichier")
        self.history.heading("status", text="Statut")
        self.history.heading("progress", text="Avancement")
        self.history.heading("date", text="Date")
        self.history.column("#0", width=220, minwidth=160)
        self.history.column("status", width=96, anchor=W)
        self.history.column("progress", width=90, anchor=W)
        self.history.column("date", width=130, anchor=W)
        self.history.tag_configure(STATUS_READY, foreground="#374151")
        self.history.tag_configure(STATUS_QUEUED, foreground="#7c3aed")
        self.history.tag_configure(STATUS_CONVERTING, foreground="#0f766e")
        self.history.tag_configure(STATUS_TRANSCRIBING, foreground="#075ac9")
        self.history.tag_configure(STATUS_DONE, foreground="#15803d")
        self.history.tag_configure(STATUS_ERROR, foreground="#dc2626")
        self.history.tag_configure(STATUS_CANCELLED, foreground="#92400e")
        self.history.bind("<<TreeviewSelect>>", self._on_select_job)
        self.history.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(parent, orient=VERTICAL, command=self.history.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.history.configure(yscrollcommand=scrollbar.set)

    def _build_transcript_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        top_line = ttk.Frame(parent, style="Surface.TFrame")
        top_line.grid(row=0, column=0, sticky="ew")
        top_line.columnconfigure(0, weight=1)
        self.file_title = ttk.Label(
            top_line,
            text="Aucune transcription selectionnee",
            style="Header.TLabel",
            font=("Segoe UI", 12, "bold"),
        )
        self.file_title.grid(row=0, column=0, sticky="w")

        button_line = ttk.Frame(top_line, style="Surface.TFrame")
        button_line.grid(row=0, column=1, sticky="e")
        self.mode_selector = ttk.Combobox(
            button_line,
            state="readonly",
            width=22,
            textvariable=self.output_mode,
            values=tuple(MODE_LABELS[mode] for mode in MODE_ORDER),
        )
        self.mode_selector.pack(side=LEFT, padx=(0, 6))
        self.mode_selector.bind("<<ComboboxSelected>>", self._on_mode_change)
        ttk.Button(button_line, text="Copier", command=self._copy_transcription).pack(side=LEFT, padx=(0, 6))
        ttk.Button(button_line, text="Exporter", command=self._export_transcription).pack(side=LEFT, padx=(0, 6))
        ttk.Button(button_line, text="Dossier", command=self._open_output_folder).pack(side=LEFT)

        self.meta_label = ttk.Label(parent, text="", style="Muted.TLabel")
        self.meta_label.grid(row=1, column=0, sticky="w", pady=(8, 8))

        text_frame = ttk.Frame(parent, style="Surface.TFrame")
        text_frame.grid(row=2, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.transcript_text = tk.Text(
            text_frame,
            wrap="word",
            font=("Segoe UI", 11),
            relief="solid",
            bd=1,
            padx=14,
            pady=12,
            foreground="#111827",
            background="#ffffff",
            insertbackground="#111827",
        )
        self.transcript_text.grid(row=0, column=0, sticky="nsew")
        transcript_scroll = ttk.Scrollbar(text_frame, orient=VERTICAL, command=self.transcript_text.yview)
        transcript_scroll.grid(row=0, column=1, sticky="ns")
        self.transcript_text.configure(yscrollcommand=transcript_scroll.set)
        self._set_text("Ajoutez un ou plusieurs fichiers, puis lancez la transcription quand vous voulez.")

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.root, style="Surface.TFrame", padding=(16, 10))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        footer.columnconfigure(3, weight=2)

        ttk.Label(footer, text="Processus", style="Header.TLabel", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 18)
        )
        self.process_label = ttk.Label(footer, text="Pret", style="Status.TLabel")
        self.process_label.grid(row=1, column=0, sticky="w", padx=(0, 18))

        ttk.Label(footer, text="Progression", style="Header.TLabel", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w"
        )
        progress_line = ttk.Frame(footer, style="Surface.TFrame")
        progress_line.grid(row=1, column=1, sticky="ew", padx=(0, 24))
        progress_line.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_line, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.progress_value = ttk.Label(progress_line, text="0%", style="Muted.TLabel")
        self.progress_value.grid(row=0, column=1)

        ttk.Label(footer, text="Message", style="Header.TLabel", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=3, sticky="w"
        )
        self.message_label = ttk.Label(footer, text="Aucun message.", style="Muted.TLabel", wraplength=520)
        self.message_label.grid(row=1, column=3, sticky="w")

    def _load_history(self) -> None:
        for record in self.store.latest():
            self.jobs[record.id] = record
            self._upsert_tree_record(record)
        self._refresh_queue_count()

    def _show_preflight(self) -> None:
        report = check_environment(self.config)
        self._refresh_environment_labels()
        if report.ok_for_local_run:
            self._set_message("Pret. Traitement local uniquement.", error=False)
            return
        if report.messages:
            self._set_message(" | ".join(report.messages), error=True)

    def _start_worker(self) -> None:
        thread = threading.Thread(target=self._worker_loop, daemon=True)
        thread.start()

    def _worker_loop(self) -> None:
        while True:
            record = self.worker_queue.get()
            if record is None:
                return
            try:
                current = self.jobs.get(record.id, record)
                if current.status != STATUS_QUEUED or current.id not in self.queued_ids:
                    continue
                token = CancellationToken()
                self.current_cancel_token = token
                self.active_job_id = current.id
                self.pipeline.process_record(
                    current,
                    allow_model_download=self.allow_model_download,
                    progress_callback=lambda item, msg: self.ui_queue.put((item, msg)),
                    cancel_token=token,
                )
            finally:
                self.queued_ids.discard(record.id)
                if self.active_job_id == record.id:
                    self.active_job_id = None
                    self.current_cancel_token = None
                self.worker_queue.task_done()

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                record, message = self.ui_queue.get_nowait()
                self.jobs[record.id] = record
                self._upsert_tree_record(record)
                self._set_current_process(record, message)
                if self._selected_record_id() == record.id:
                    self._display_record(record)
        except queue.Empty:
            pass
        self.root.after(200, self._drain_ui_queue)

    def _upsert_tree_record(self, record: JobRecord) -> None:
        values = (
            record.status_label,
            self._progress_text(record),
            record.updated_at.replace("T", " ").replace("Z", ""),
        )
        tags = (record.status,)
        if record.id in self.tree_items:
            item_id = self.tree_items[record.id]
            self.history.item(item_id, text=record.source_name, values=values, tags=tags)
        else:
            item_id = self.history.insert("", 0, text=record.source_name, values=values, tags=tags)
            self.tree_items[record.id] = item_id
        self._refresh_queue_count()

    def _progress_text(self, record: JobRecord) -> str:
        if record.status == STATUS_DONE:
            return "100%"
        if record.status == STATUS_ERROR:
            return "Erreur"
        if record.status == STATUS_CANCELLED:
            return "Annule"
        if record.progress:
            return f"{record.progress}%"
        return "-"

    def _refresh_queue_count(self) -> None:
        total = len(self.jobs)
        pending = len([record for record in self.jobs.values() if record.status in RETRYABLE_STATUSES])
        running = len([record for record in self.jobs.values() if record.status in RUNNING_STATUSES])
        parts = [f"{total} element" + ("" if total <= 1 else "s")]
        if pending:
            parts.append(f"{pending} pret" + ("" if pending <= 1 else "s"))
        if running:
            parts.append(f"{running} en cours")
        self.queue_count_label.configure(text=" | ".join(parts))

    def _set_current_process(self, record: JobRecord, message: str) -> None:
        self.process_label.configure(text=f"{record.status_label} - {record.source_name}")
        self._set_progress(record)
        self._set_message(message or record.error or record.message, error=record.status == STATUS_ERROR)
        if record.status == STATUS_DONE:
            self._select_record(record.id)

    def _set_progress(self, record: JobRecord) -> None:
        value = max(0, min(100, int(record.progress or 0)))
        self.progress.configure(value=value)
        suffix = ""
        if record.eta_seconds is not None and record.status in RUNNING_STATUSES:
            suffix = f" | {format_eta(record.eta_seconds)}"
        elif record.progress_detail:
            detail = record.progress_detail
            if len(detail) > 36:
                detail = detail[:33].rstrip() + "..."
            suffix = f" | {detail}"
        self.progress_value.configure(text=f"{value}%{suffix}")

    def _set_message(self, message: str, error: bool = False) -> None:
        self.message_label.configure(
            text=message or "Aucun message.",
            foreground="#dc2626" if error else "#6b7280",
        )

    def _refresh_environment_labels(self) -> None:
        report = check_environment(self.config)
        if report.ok_for_local_run:
            self.model_label.configure(
                text=(
                    f"{report.backend_label} | Modele {self.config.model_size} local | "
                    f"{PROFILE_LABELS.get(self.config.transcription_profile, self.config.transcription_profile)}"
                )
            )
        else:
            self.model_label.configure(text=f"{report.backend_label} | {self.config.model_status_label(report.backend_id)}")

    def _selected_record_id(self) -> str | None:
        selection = self.history.selection()
        if not selection:
            return None
        selected_item = selection[0]
        for job_id, item_id in self.tree_items.items():
            if item_id == selected_item:
                return job_id
        return None

    def _selected_record(self) -> JobRecord | None:
        record_id = self._selected_record_id()
        return self.jobs.get(record_id) if record_id else None

    def _selected_records(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        selected_items = set(self.history.selection())
        for job_id, item_id in self.tree_items.items():
            if item_id in selected_items and job_id in self.jobs:
                records.append(self.jobs[job_id])
        return records

    def _select_record(self, job_id: str) -> None:
        item_id = self.tree_items.get(job_id)
        if item_id:
            self.history.selection_set(item_id)
            self.history.focus(item_id)
            self._display_record(self.jobs[job_id])

    def _on_select_job(self, _event=None) -> None:
        record = self._selected_record()
        if record:
            self._display_record(record)

    def _display_record(self, record: JobRecord) -> None:
        self.file_title.configure(text=record.source_name)
        meta_parts = [
            f"Statut : {record.status_label}",
            f"Moteur : {BACKEND_LABELS.get(record.backend or resolve_backend_id(self.config), record.backend or resolve_backend_id(self.config))}",
            f"Modele : {record.model_size or self.config.model_size}",
            f"Profil : {PROFILE_LABELS.get(record.transcription_profile or self.config.transcription_profile, record.transcription_profile or self.config.transcription_profile)}",
        ]
        if record.duration_seconds:
            minutes = int(record.duration_seconds // 60)
            seconds = int(record.duration_seconds % 60)
            meta_parts.append(f"Duree : {minutes:02d}:{seconds:02d}")
        if record.progress_detail and record.status in RUNNING_STATUSES:
            meta_parts.append(record.progress_detail)
        self.meta_label.configure(text=" | ".join(meta_parts))

        selected_path = self._record_output_path(record)
        if selected_path and selected_path.exists():
            self._set_text(self._read_record_output(record))
        elif record.error:
            self._set_text(record.error)
        elif record.status in {STATUS_READY, STATUS_CANCELLED}:
            self._set_text("Fichier ajoute. Lancez la transcription pour generer le texte.")
        else:
            self._set_text("La transcription n'est pas encore disponible.")

    def _set_text(self, text: str) -> None:
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", END)
        self.transcript_text.insert("1.0", text)
        self.transcript_text.configure(state="disabled")

    def _ensure_transcription_ready(self, records: list[JobRecord]) -> bool:
        if self.installing_dependency:
            self._set_message("Installation ou telechargement en cours. Attendez la fin avant de transcrire.", error=True)
            return False

        report = check_environment(self.config)
        missing = []
        if not report.backend_compatible:
            missing.append(f"{report.backend_label} non compatible")
        if not report.backend_dependency_ok:
            missing.append(report.backend_label)
        if missing:
            self._ask_open_settings(
                "Installation requise",
                "WisperAuto ne peut pas lancer la transcription car il manque :\n\n"
                + "\n".join(f"- {item}" for item in missing),
            )
            return False

        if not self.config.local_model_path(report.backend_id):
            self._ask_open_settings(
                "Modele absent",
                f"Aucun modele local n'a ete trouve pour {report.backend_label}.\n\n"
                "Telechargez le modele depuis Parametres avant de transcrire.",
            )
            return False

        needs_ffmpeg = any(self.pipeline.needs_conversion(self._record_source_path(record) or Path(record.source_path)) for record in records)
        if needs_ffmpeg and not report.ffmpeg_ok:
            self._ask_open_settings(
                "FFmpeg requis",
                "Un ou plusieurs fichiers selectionnes doivent etre convertis avant transcription, mais FFmpeg est absent.",
            )
            return False

        return True

    def _ask_open_settings(self, title: str, message: str) -> None:
        open_settings = messagebox.askyesno(
            title,
            message + "\n\nOuvrir les parametres maintenant ?",
        )
        if open_settings:
            self._show_settings()
        self._set_message(message.replace("\n", " "), error=True)

    def _on_mode_change(self, _event=None) -> None:
        selected = MODE_BY_LABEL.get(self.output_mode.get(), MODE_SMART)
        self.config = replace(self.config, output_mode=selected)
        self.config.save_user_settings()
        self.pipeline.config = self.config
        self._refresh_selected_record()

    def _refresh_selected_record(self) -> None:
        record = self._selected_record()
        if record:
            self._display_record(record)

    def _import_audio(self) -> None:
        extensions = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Ajouter des fichiers audio",
            filetypes=[("Fichiers audio", extensions), ("Tous les fichiers", "*.*")],
        )
        if not paths:
            return

        imported: list[JobRecord] = []
        for selected in paths:
            try:
                record = self.pipeline.import_audio_job(Path(selected))
            except Exception as exc:
                self._set_message(str(exc), error=True)
                continue
            self.jobs[record.id] = record
            self._upsert_tree_record(record)
            imported.append(record)

        if imported:
            self._select_record(imported[0].id)
            count = len(imported)
            self._set_message(
                f"{count} fichier" + ("" if count <= 1 else "s") + " ajoute" + ("" if count <= 1 else "s") + ". Lancez la transcription quand vous etes pret.",
                error=False,
            )

    def _transcribe_selected(self) -> None:
        records = self._selected_records()
        if not records:
            self._set_message("Selectionnez au moins un fichier a transcrire.", error=True)
            return
        self._enqueue_records(records)

    def _transcribe_all(self) -> None:
        records = [record for record in self.jobs.values() if record.status in RETRYABLE_STATUSES]
        if not records:
            self._set_message("Aucun fichier pret a transcrire.", error=True)
            return
        self._enqueue_records(records)

    def _enqueue_records(self, records: list[JobRecord]) -> None:
        candidates: list[JobRecord] = []
        for record in records:
            if record.status == STATUS_DONE:
                continue
            if record.status in RUNNING_STATUSES:
                continue
            source_path = self._record_source_path(record)
            if not source_path:
                self._set_message(f"Fichier source introuvable : {record.source_name}", error=True)
                continue
            if source_path != Path(record.source_path):
                record = self.store.update(record, source_path=str(source_path), failed_path="")
                self.jobs[record.id] = record
            candidates.append(record)

        if not candidates:
            self._set_message("Aucun fichier selectionne ne peut etre lance.", error=True)
            return
        if not self._ensure_transcription_ready(candidates):
            return

        for record in candidates:
            queued = self.store.update(
                record,
                status=STATUS_QUEUED,
                progress=max(1, record.progress),
                phase="queued",
                message="En attente de transcription.",
                progress_detail="",
                eta_seconds=None,
                error="",
            )
            self.jobs[queued.id] = queued
            self.queued_ids.add(queued.id)
            self._upsert_tree_record(queued)
            self.worker_queue.put(queued)
        self._set_message(f"{len(candidates)} fichier(s) ajoute(s) au batch.", error=False)

    def _cancel_selected_or_active(self) -> None:
        records = self._selected_records()
        cancelled = 0
        for record in records:
            if record.id == self.active_job_id and self.current_cancel_token:
                self.current_cancel_token.cancel()
                cancelled += 1
            elif record.id in self.queued_ids and record.status == STATUS_QUEUED:
                updated = self.store.update(
                    record,
                    status=STATUS_CANCELLED,
                    phase="cancelled",
                    message="Annule avant lancement.",
                    progress_detail="",
                    eta_seconds=None,
                )
                self.queued_ids.discard(record.id)
                self.jobs[record.id] = updated
                self._upsert_tree_record(updated)
                cancelled += 1

        if cancelled:
            self._set_message("Annulation demandee.", error=False)
            return

        if self.current_cancel_token:
            self.current_cancel_token.cancel()
            self._set_message("Annulation du traitement en cours demandee.", error=False)
        else:
            self._set_message("Aucun traitement selectionne a annuler.", error=True)

    def _delete_selected(self) -> None:
        records = self._selected_records()
        if not records:
            self._set_message("Selectionnez une retranscription a supprimer.", error=True)
            return

        locked = [record for record in records if record.status in RUNNING_STATUSES or record.id == self.active_job_id]
        if locked:
            self._set_message("Annulez ou laissez terminer le traitement avant de supprimer.", error=True)
            return

        confirmed = messagebox.askyesno(
            "Supprimer",
            "Supprimer l'entree d'historique et les fichiers .txt generes ?\n\n"
            "Les audios archives ou en erreur seront conserves.",
        )
        if not confirmed:
            return

        deleted_count = 0
        for record in records:
            self._delete_record_files(record)
            self.store.delete(record.id)
            item_id = self.tree_items.pop(record.id, None)
            if item_id:
                self.history.delete(item_id)
            self.jobs.pop(record.id, None)
            self.queued_ids.discard(record.id)
            deleted_count += 1

        self._refresh_queue_count()
        self._set_text("Aucune transcription selectionnee.")
        self.file_title.configure(text="Aucune transcription selectionnee")
        self.meta_label.configure(text="")
        self._set_message(f"{deleted_count} entree(s) supprimee(s).", error=False)

    def _delete_record_files(self, record: JobRecord) -> None:
        paths = set(record.outputs.values()) if record.outputs else set()
        paths.update(path for path in (record.transcript_path, record.raw_transcript_path) if path)
        for path_text in paths:
            path = Path(path_text)
            if path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass

        source_path = Path(record.source_path)
        if source_path.exists() and self._is_relative_to(source_path, self.config.inbox_dir):
            try:
                source_path.unlink()
            except OSError:
                pass

    def _record_source_path(self, record: JobRecord) -> Path | None:
        source_path = Path(record.source_path)
        if source_path.exists():
            return source_path
        failed_path = Path(record.failed_path) if record.failed_path else None
        if failed_path and failed_path.exists():
            return failed_path
        return None

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def _copy_transcription(self) -> None:
        content = self._current_transcript_content()
        if not content:
            self._set_message("Aucune transcription a copier.", error=True)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._set_message("Transcription copiee dans le presse-papiers.", error=False)

    def _export_transcription(self) -> None:
        record = self._selected_record()
        output_path = self._record_output_path(record) if record else None
        if not record or not output_path or not output_path.exists():
            self._set_message("Aucune transcription a exporter.", error=True)
            return
        destination = filedialog.asksaveasfilename(
            title="Exporter la transcription",
            defaultextension=".txt",
            initialfile=output_path.name,
            filetypes=[("Texte", "*.txt")],
        )
        if not destination:
            return
        shutil.copyfile(output_path, destination)
        self._set_message(f"Transcription exportee : {destination}", error=False)

    def _open_output_folder(self) -> None:
        record = self._selected_record()
        selected_path = self._record_output_path(record) if record else None
        target = selected_path.parent if selected_path else self.config.outbox_dir
        try:
            if sys.platform.startswith("win"):
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            self._set_message(f"Impossible d'ouvrir le dossier : {exc}", error=True)

    def _show_settings(self) -> None:
        if self.settings_dialog and self.settings_dialog.exists():
            self.settings_dialog.focus()
            return
        self.settings_dialog = SettingsDialog(self)

    def _current_transcript_content(self) -> str:
        record = self._selected_record()
        path = self._record_output_path(record) if record else None
        if not path or not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _record_output_path(self, record: JobRecord | None) -> Path | None:
        if not record:
            return None
        selected = MODE_BY_LABEL.get(self.output_mode.get(), self.output_mode.get())
        path = record.outputs.get(selected) if record.outputs else ""
        if not path:
            if selected == MODE_RAW and record.raw_transcript_path:
                path = record.raw_transcript_path
            else:
                path = record.transcript_path
        return Path(path) if path else None

    def _read_record_output(self, record: JobRecord) -> str:
        path = self._record_output_path(record)
        if not path or not path.exists():
            return "La transcription n'est pas encore disponible dans ce mode."
        return path.read_text(encoding="utf-8")


def run_app(config: AppConfig | None = None) -> None:
    config = config or AppConfig.from_env()
    config.ensure_directories()
    root = tk.Tk()
    WisperAutoApp(root, config)
    root.mainloop()


class SettingsDialog:
    def __init__(self, app: WisperAutoApp):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Parametres et installation")
        self.window.geometry("760x540")
        self.window.minsize(680, 460)
        self.window.transient(app.root)
        self.window.configure(bg="#ffffff")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar()
        self.python_var = tk.StringVar()
        self.ffmpeg_var = tk.StringVar()
        self.ffprobe_var = tk.StringVar()
        self.whisper_var = tk.StringVar()
        self.mlx_var = tk.StringVar()
        self.whisper_cpp_var = tk.StringVar()
        self.active_backend_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.backend_choice = tk.StringVar(
            value=BACKEND_LABELS.get(app.config.backend, BACKEND_LABELS[BACKEND_AUTO])
        )
        self.model_choice = tk.StringVar(value=app.config.model_size)
        self.profile_choice = tk.StringVar(
            value=PROFILE_LABELS.get(app.config.transcription_profile, PROFILE_LABELS[PROFILE_FAST])
        )
        self.progress_running = False
        self.cancel_token: CancellationToken | None = None

        self._build()
        self.refresh()

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def focus(self) -> None:
        self.window.lift()
        self.window.focus_force()

    def close(self) -> None:
        if self.app.installing_dependency:
            messagebox.showinfo(
                "Installation en cours",
                "Utilisez Annuler ou attendez la fin avant de fermer cette fenetre.",
                parent=self.window,
            )
            return
        self.app.settings_dialog = None
        self.window.destroy()

    def _build(self) -> None:
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(3, weight=1)

        header = ttk.Frame(self.window, style="Surface.TFrame", padding=(16, 12))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            header,
            text="Parametres et installation",
            style="Header.TLabel",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Les audios restent locaux. Les installations et modeles utilisent Internet apres confirmation.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        info = ttk.Frame(self.window, style="Surface.TFrame", padding=(16, 6))
        info.grid(row=1, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)

        rows = [
            ("Dossier", str(self.app.config.home)),
            ("Outbox", str(self.app.config.outbox_dir)),
            ("Langue", self.app.config.language),
        ]
        for row, (label, value) in enumerate(rows):
            ttk.Label(info, text=label, style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(info, text=value, style="Muted.TLabel").grid(row=row, column=1, sticky="w", pady=2)

        row = len(rows)
        ttk.Label(info, text="Moteur", style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        backend_selector = ttk.Combobox(
            info,
            state="readonly",
            width=18,
            textvariable=self.backend_choice,
            values=tuple(BACKEND_LABELS[backend] for backend in BACKEND_ORDER),
        )
        backend_selector.grid(row=row, column=1, sticky="w", pady=2)
        backend_selector.bind("<<ComboboxSelected>>", lambda _event: self.update_backend_choice())

        row += 1
        ttk.Label(info, text="Modele Whisper", style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        model_selector = ttk.Combobox(
            info,
            state="readonly",
            width=18,
            textvariable=self.model_choice,
            values=MODEL_CHOICES,
        )
        model_selector.grid(row=row, column=1, sticky="w", pady=2)
        model_selector.bind("<<ComboboxSelected>>", lambda _event: self.update_model_choice())

        row += 1
        ttk.Label(info, text="Profil", style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        profile_selector = ttk.Combobox(
            info,
            state="readonly",
            width=18,
            textvariable=self.profile_choice,
            values=tuple(PROFILE_LABELS.values()),
        )
        profile_selector.grid(row=row, column=1, sticky="w", pady=2)
        profile_selector.bind("<<ComboboxSelected>>", lambda _event: self.update_profile_choice())

        status = ttk.Frame(self.window, style="Surface.TFrame", padding=(16, 6))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)

        status_rows = [
            ("Python", self.python_var),
            ("Moteur actif", self.active_backend_var),
            ("FFmpeg", self.ffmpeg_var),
            ("ffprobe", self.ffprobe_var),
            ("faster-whisper", self.whisper_var),
            ("MLX Mac", self.mlx_var),
            ("whisper.cpp", self.whisper_cpp_var),
            ("Modele moteur", self.model_var),
        ]
        for row, (label, variable) in enumerate(status_rows):
            ttk.Label(status, text=label, style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(status, textvariable=variable, style="Muted.TLabel").grid(
                row=row, column=1, sticky="w", pady=3
            )

        actions = ttk.Frame(status, style="Surface.TFrame")
        actions.grid(row=0, column=2, rowspan=7, sticky="ne", padx=(16, 0))
        self.ffmpeg_button = ttk.Button(actions, text="Installer FFmpeg", command=self.install_ffmpeg)
        self.ffmpeg_button.pack(fill="x", pady=(0, 7))
        self.backend_button = ttk.Button(actions, text="Installer moteur", command=self.install_selected_backend)
        self.backend_button.pack(fill="x", pady=(0, 7))
        self.model_button = ttk.Button(actions, text="Telecharger modele moteur", command=self.install_model)
        self.model_button.pack(fill="x", pady=(0, 7))
        self.benchmark_button = ttk.Button(actions, text="Tester performances", command=self.run_benchmark)
        self.benchmark_button.pack(fill="x", pady=(0, 7))
        self.refresh_button = ttk.Button(actions, text="Reverifier", command=self.refresh)
        self.refresh_button.pack(fill="x")

        log_frame = ttk.Frame(self.window, style="Surface.TFrame", padding=(16, 6))
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        ttk.Label(log_frame, text="Journal", style="Header.TLabel", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.log_text = tk.Text(log_frame, height=10, wrap="word", font=("Consolas", 9), relief="solid", bd=1, padx=10, pady=8)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(self.window, style="Surface.TFrame", padding=(16, 10))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.install_progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.install_progress.grid(row=0, column=1, sticky="ew", padx=14)
        self.cancel_button = ttk.Button(footer, text="Annuler", command=self.cancel_install, state="disabled")
        self.cancel_button.grid(row=0, column=2, sticky="e", padx=(0, 8))
        ttk.Button(footer, text="Fermer", command=self.close).grid(row=0, column=3, sticky="e")

    def refresh(self) -> None:
        report = check_environment(self.app.config)
        selected_backend = self._selected_backend_for_actions()
        selected_health = backend_health(self.app.config, selected_backend)
        local_model = self.app.config.local_model_path(selected_health.backend_id)
        self.backend_choice.set(BACKEND_LABELS.get(self.app.config.backend, BACKEND_LABELS[BACKEND_AUTO]))
        self.model_choice.set(self.app.config.model_size)
        self.profile_choice.set(PROFILE_LABELS.get(self.app.config.transcription_profile, PROFILE_LABELS[PROFILE_FAST]))
        active_parts = [f"{report.backend_label} ({report.device}, {report.compute_type})"]
        if report.backend_id == "faster-whisper":
            active_parts.append(f"{report.cpu_threads} threads")
            if report.batch_size > 1:
                active_parts.append(f"batch {report.batch_size}")
        elif report.backend_id == "whisper.cpp":
            active_parts.append(f"{report.cpu_threads} threads")
        active_parts.append(f"VAD {report.vad_silence_ms}ms")
        self.active_backend_var.set(" | ".join(active_parts))
        self.python_var.set(self._python_status_label())
        self.ffmpeg_var.set("OK" if report.ffmpeg_ok else "Manquant")
        self.ffprobe_var.set("OK" if report.ffprobe_ok else "Manquant")
        self.whisper_var.set("OK" if report.faster_whisper_ok else "Manquant")
        self.mlx_var.set("OK" if backend_health(self.app.config, "mlx-whisper").dependency_ok else "Manquant ou incompatible")
        self.whisper_cpp_var.set("OK" if backend_health(self.app.config, "whisper.cpp").dependency_ok else "Binaire introuvable")
        self.model_var.set(f"Local : {local_model}" if local_model else "Absent - telechargement requis")
        self.backend_button.configure(text=f"Installer {selected_health.label}")
        self.model_button.configure(text=f"Telecharger modele {selected_health.label}")
        if report.messages:
            self.status_var.set(" | ".join(report.messages))
        else:
            self.status_var.set("Environnement pret.")
        self.app._show_preflight()

    def _python_status_label(self) -> str:
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        prefix = "Conda detecte, venv projet recommande - " if os.environ.get("CONDA_PREFIX") else ""
        if sys.version_info[:2] in {(3, 11), (3, 12)}:
            return f"{prefix}OK Python {version}"
        return f"{prefix}A verifier : Python {version}, cible 3.11/3.12"

    def _selected_backend_for_actions(self) -> str:
        selected = BACKEND_BY_LABEL.get(self.backend_choice.get(), self.app.config.backend)
        if selected == BACKEND_AUTO:
            return resolve_backend_id(self.app.config)
        return selected

    def update_backend_choice(self) -> None:
        selected = BACKEND_BY_LABEL.get(self.backend_choice.get(), BACKEND_AUTO)
        if selected == self.app.config.backend:
            return
        if self.app.installing_dependency:
            self.backend_choice.set(BACKEND_LABELS.get(self.app.config.backend, BACKEND_LABELS[BACKEND_AUTO]))
            messagebox.showinfo("Installation en cours", "Attendez la fin de l'installation avant de changer de moteur.", parent=self.window)
            return

        self.app.config = replace(self.app.config, backend=selected)
        self.app.config.ensure_directories()
        self.app.config.save_user_settings()
        self.app.pipeline = TranscriptionPipeline(self.app.config, self.app.store)
        self.app._refresh_environment_labels()
        self.log(f"Moteur selectionne : {BACKEND_LABELS.get(selected, selected)}")
        self.refresh()

    def update_model_choice(self) -> None:
        selected = self.model_choice.get()
        if selected == self.app.config.model_size:
            return
        if self.app.installing_dependency:
            self.model_choice.set(self.app.config.model_size)
            messagebox.showinfo("Installation en cours", "Attendez la fin de l'installation avant de changer de modele.", parent=self.window)
            return

        self.app.config = replace(self.app.config, model_size=selected)
        self.app.config.ensure_directories()
        self.app.config.save_user_settings()
        self.app.pipeline = TranscriptionPipeline(self.app.config, self.app.store)
        self.app._refresh_environment_labels()
        self.log(f"Modele selectionne : {selected}")
        self.refresh()

    def update_profile_choice(self) -> None:
        selected_label = self.profile_choice.get()
        selected = PROFILE_BY_LABEL.get(selected_label, PROFILE_FAST)
        if selected == self.app.config.transcription_profile:
            return
        if self.app.installing_dependency:
            self.profile_choice.set(PROFILE_LABELS.get(self.app.config.transcription_profile, PROFILE_LABELS[PROFILE_FAST]))
            return

        self.app.config = replace(self.app.config, transcription_profile=selected)
        self.app.config.save_user_settings()
        self.app.pipeline = TranscriptionPipeline(self.app.config, self.app.store)
        self.app._refresh_environment_labels()
        self.log(f"Profil selectionne : {selected_label}")
        self.refresh()

    def install_ffmpeg(self) -> None:
        try:
            plan = ffmpeg_plan()
        except InstallUnavailableError as exc:
            messagebox.showerror("Installation FFmpeg", str(exc), parent=self.window)
            self.log(str(exc))
            return
        self._confirm_and_run(plan)

    def install_selected_backend(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        backend = self._selected_backend_for_actions()
        try:
            plan = backend_install_plan(backend, project_root)
        except InstallUnavailableError as exc:
            messagebox.showerror("Installation moteur", str(exc), parent=self.window)
            self.log(str(exc))
            return
        self._confirm_and_run(plan)

    def install_model(self) -> None:
        backend = self._selected_backend_for_actions()
        health = backend_health(self.app.config, backend)
        if not health.dependency_ok:
            messagebox.showerror(
                "Modele",
                f"Installez d'abord {health.label} avant de telecharger son modele.",
                parent=self.window,
            )
            return

        local_model = self.app.config.local_model_path(health.backend_id)
        if local_model:
            messagebox.showinfo("Modele", f"Modele deja disponible :\n{local_model}", parent=self.window)
            return

        if self.app.installing_dependency:
            messagebox.showinfo("Installation en cours", "Une installation est deja en cours.", parent=self.window)
            return

        confirmed = messagebox.askyesno(
            "Telecharger le modele",
            f"WisperAuto va telecharger le modele {self.app.config.model_size} pour {health.label} dans :\n\n"
            f"{self.app.config.backend_model_dir(health.backend_id)}\n\n"
            "Cette operation utilise Internet. Le journal affichera le debit, la taille visible et une ETA quand possible. Continuer ?",
            parent=self.window,
        )
        if not confirmed:
            return

        self.app.installing_dependency = True
        self.cancel_token = CancellationToken()
        self._set_install_buttons("disabled")
        self._start_progress()
        self.status_var.set(f"Telechargement du modele {self.app.config.model_size} pour {health.label} en cours...")
        thread = threading.Thread(target=self._run_model_download, args=(health.backend_id,), daemon=True)
        thread.start()

    def run_benchmark(self) -> None:
        if self.app.installing_dependency:
            messagebox.showinfo("Operation en cours", "Une operation est deja en cours.", parent=self.window)
            return

        extensions = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
        selected = filedialog.askopenfilename(
            title="Choisir un fichier audio pour le benchmark local",
            filetypes=[("Fichiers audio", extensions), ("Tous les fichiers", "*.*")],
            parent=self.window,
        )
        if not selected:
            return

        confirmed = messagebox.askyesno(
            "Benchmark local",
            "WisperAuto va creer un extrait temporaire local et tester les moteurs qui ont deja un modele local.\n\n"
            "Aucun audio n'est envoye en ligne. Continuer ?",
            parent=self.window,
        )
        if not confirmed:
            return

        self.app.installing_dependency = True
        self.cancel_token = CancellationToken()
        self._set_install_buttons("disabled")
        self._start_progress()
        self.status_var.set("Benchmark local en cours...")
        self.log(f"Benchmark local : {selected}")
        thread = threading.Thread(target=self._run_benchmark, args=(Path(selected),), daemon=True)
        thread.start()

    def _confirm_and_run(self, plan: InstallPlan) -> None:
        if self.app.installing_dependency:
            messagebox.showinfo("Installation en cours", "Une installation est deja en cours.", parent=self.window)
            return

        commands = "\n".join(" ".join(command) for command in plan.commands)
        confirmed = messagebox.askyesno(
            f"Installer {plan.name}",
            f"WisperAuto va executer :\n\n{commands}\n\n"
            "Cette operation peut utiliser Internet et modifier l'environnement local. Continuer ?",
            parent=self.window,
        )
        if not confirmed:
            return

        self.app.installing_dependency = True
        self.cancel_token = CancellationToken()
        self._set_install_buttons("disabled")
        self._start_progress()
        self.status_var.set(f"Installation de {plan.name} en cours...")
        thread = threading.Thread(target=self._run_install, args=(plan,), daemon=True)
        thread.start()

    def cancel_install(self) -> None:
        if self.cancel_token:
            self.cancel_token.cancel()
            self.status_var.set("Annulation demandee...")
            self.log("Annulation demandee par l'utilisateur.")

    def _run_install(self, plan: InstallPlan) -> None:
        try:
            run_install_plan(
                plan,
                logger=lambda message: self.window.after(0, self.log, message),
                cancel_token=self.cancel_token,
            )
        except OperationCancelledError as exc:
            self.window.after(0, self._install_finished, plan.name, False, str(exc), True)
            return
        except Exception as exc:
            self.window.after(0, self._install_finished, plan.name, False, str(exc), False)
            return
        self.window.after(0, self._install_finished, plan.name, True, "", False)

    def _run_model_download(self, backend: str) -> None:
        try:
            download_model(
                self.app.config,
                backend=backend,
                logger=lambda message: self.window.after(0, self.log, message),
                cancel_token=self.cancel_token,
            )
        except OperationCancelledError as exc:
            self.window.after(0, self._install_finished, "Modele", False, str(exc), True)
            return
        except Exception as exc:
            self.window.after(0, self._install_finished, "Modele", False, str(exc), False)
            return
        self.window.after(0, self._install_finished, "Modele", True, "", False)

    def _run_benchmark(self, path: Path) -> None:
        try:
            results = benchmark_backends(
                self.app.config,
                path,
                logger=lambda message: self.window.after(0, self.log, message),
                cancel_token=self.cancel_token,
            )
        except OperationCancelledError as exc:
            self.window.after(0, self._benchmark_finished, False, str(exc), True, "")
            return
        except Exception as exc:
            self.window.after(0, self._benchmark_finished, False, str(exc), False, "")
            return
        self.window.after(0, self._benchmark_finished, True, "", False, recommendation_text(results))

    def _benchmark_finished(self, ok: bool, error: str, cancelled: bool, recommendation: str) -> None:
        self.app.installing_dependency = False
        self.cancel_token = None
        self._set_install_buttons("normal")
        self._stop_progress()
        if ok:
            self.log(recommendation)
            self.status_var.set("Benchmark termine.")
        elif cancelled:
            self.log("Benchmark annule.")
            self.status_var.set("Benchmark annule.")
        else:
            self.log(f"Erreur benchmark : {error}")
            self.status_var.set("Erreur benchmark.")
            messagebox.showerror("Benchmark", error, parent=self.window)
        self.refresh()

    def _install_finished(self, name: str, ok: bool, error: str, cancelled: bool) -> None:
        self.app.installing_dependency = False
        self.cancel_token = None
        self._set_install_buttons("normal")
        self._stop_progress()
        if ok:
            if name == "Modele":
                self.app.allow_model_download = True
            self.log(f"{name} installe. Reverifiez l'environnement.")
            self.status_var.set(f"{name} installe.")
        elif cancelled:
            self.log(f"{name} annule.")
            self.status_var.set(f"{name} annule.")
        else:
            self.log(f"Erreur installation {name}: {error}")
            self.status_var.set(f"Erreur installation {name}.")
            messagebox.showerror(f"Installation {name}", error, parent=self.window)
        self.refresh()

    def _set_install_buttons(self, state: str) -> None:
        self.ffmpeg_button.configure(state=state)
        self.backend_button.configure(state=state)
        self.model_button.configure(state=state)
        self.benchmark_button.configure(state=state)
        self.refresh_button.configure(state=state)
        self.cancel_button.configure(state="normal" if state == "disabled" else "disabled")

    def _start_progress(self) -> None:
        if not self.progress_running:
            self.install_progress.start(12)
            self.progress_running = True

    def _stop_progress(self) -> None:
        if self.progress_running:
            self.install_progress.stop()
            self.progress_running = False

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self.log_text.configure(state="disabled")
