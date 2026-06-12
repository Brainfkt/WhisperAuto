"""LLM-first local post-processing for French legal dictation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import PostProcessUnavailableError
from .postprocess_llm import PostProcessProgress


MODE_RAW = "raw"
MODE_CLEANED = "cleaned"
MODE_SMART = "smart"
MODE_REPORT = "report"

MODE_LABELS = {
    MODE_RAW: "Transcription brute",
    MODE_SMART: "Transcription intelligente",
    MODE_REPORT: "Compte rendu structure",
}


@dataclass
class PostProcessResult:
    outputs: dict[str, str]
    actions: list[str]


class PostProcessor:
    def __init__(
        self,
        commands_path: Path,
        config=None,
        postprocess_engine: str | None = None,
        llm_provider=None,
    ):
        self.commands_path = commands_path
        self.config = config
        self.postprocess_engine = postprocess_engine or getattr(config, "postprocess_engine", "")
        self.llm_provider = llm_provider
        self.actions: list[str] = []

    def build_outputs(
        self,
        raw_text: str,
        *,
        progress_callback: Callable[[PostProcessProgress], None] | None = None,
        cancel_token=None,
    ) -> PostProcessResult:
        self.actions = []
        raw = self._with_final_newline(raw_text)
        smart = self.process_smart(raw_text, progress_callback=progress_callback, cancel_token=cancel_token)
        report = self.build_report(smart)
        return PostProcessResult(
            outputs={
                MODE_RAW: raw,
                MODE_SMART: smart,
                MODE_REPORT: report,
            },
            actions=list(self.actions),
        )

    def process_smart(
        self,
        raw_text: str,
        *,
        progress_callback: Callable[[PostProcessProgress], None] | None = None,
        cancel_token=None,
    ) -> str:
        if self.config is None:
            raise PostProcessUnavailableError("configuration absente")

        from .postprocess_llm import DirectLLMPostProcessor

        engine = DirectLLMPostProcessor(self.config, provider=self.llm_provider)
        result = engine.apply(raw_text, progress_callback=progress_callback, cancel_token=cancel_token)
        self.llm_provider = engine.provider
        self.actions.extend(engine.actions)
        for warning in result.warnings:
            self.actions.append(f"Avertissement LLM : {warning}")
        return result.text

    def build_report(self, smart_text: str) -> str:
        content = smart_text.strip()
        if not content:
            return ""
        lines = [
            "Compte rendu structure",
            "",
            "Note : sortie locale derivee de la transcription intelligente, sans information ajoutee.",
            "",
            "Contenu retranscrit",
            "",
            content,
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _with_final_newline(text: str) -> str:
        stripped = text.strip()
        return stripped + ("\n" if stripped else "")
