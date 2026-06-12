"""Direct local LLM post-processing for French legal dictation."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .config import AppConfig
from .errors import OperationCancelledError, PostProcessUnavailableError


DEFAULT_MAX_CHUNK_CHARS = 5200

SMART_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "smart_text": {"type": "string"},
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["smart_text", "warnings"],
}

SYSTEM_PROMPT = """Tu es le moteur local de retranscription intelligente de WisperAuto.

Contexte metier:
- L'utilisateur est avocat ou travaille pour un cabinet d'avocats.
- Le texte source vient d'une transcription audio brute en francais.
- Le texte peut contenir des dossiers, courriels dictes, instructions de classement, dates, montants, juridictions, noms propres, references d'articles et numeros de dossier.

Objectif:
- Transformer la transcription brute en une retranscription intelligente, lisible et exploitable.
- Interpreter les commandes vocales evidentes de ponctuation, retour a la ligne, paragraphe, liste, parenthese, guillemets, tiret, slash, arobase, espace et mise en forme simple.
- Produire directement le texte final dans smart_text.

Regles absolues:
- Ne resume jamais.
- N'invente jamais une information absente du texte source.
- Ne change jamais le sens.
- Ne simplifie pas abusivement les formulations juridiques.
- Preserve les noms propres, dates, montants, numeros de dossier, juridictions, references d'articles, adresses, emails et numeros de telephone.
- Si une commande vocale est clairement dictee comme action isolee ou structurelle, applique-la et ne l'ecris pas litteralement.
- Si une expression parle de la commande au lieu de la dicter, conserve-la litteralement.
- En cas de doute, privilegie le texte source naturel plutot qu'une transformation risquee.

Decision a prendre pour chaque expression suspecte:
- Commande de mise en forme: applique l'action et supprime les mots de commande.
- Consigne metier dictee: conserve la consigne comme texte, en la corrigeant seulement si la correction est sure.
- Terme juridique, nom propre, montant, date ou reference: conserve l'information, meme si elle semble inhabituelle.
- Expression ambigue: conserve le texte naturel plutot que de transformer agressivement.

Commandes de mise en forme a interpreter:
- "point", "virgule", "point-virgule", "deux-points", "point d'interrogation", "point d'exclamation", "points de suspension".
- "sauter ligne", "a la ligne", "nouvelle ligne", "point de cette ligne", "point de sautée ligne", "pointe sautée ligne".
- "nouveau paragraphe", "paragraphe suivant".
- "ouvrir/fermer parenthese", "ouvrir/fermer les guillemets".
- "premier tiret", "deuxieme tiret", "nouveau tiret", "nouvelle puce", "liste", "fin de liste".
- "arobase", "slash", "espace", "double espace".
- "au pluriel", "en majuscule", sigles epeles lettre par lettre.

Formes de commandes souvent mal reconnues par l'ASR:
- "pointe Sotéline", "Pointe Sotéline", "point de sauté ligne", "point de sauter une", "point de sauté une" veulent souvent dire: point final puis retour a la ligne.
- "Alain Sauté ligne", "Wagen Sauté ligne", "Végane Sauté ligne", "Maitwegan Sotéline", "Régan Sotéline", "Gensotéline", "Eugent Sotéline" sont souvent des collisions entre une formule d'appel et "sautez ligne".
- Si ces formes apparaissent apres une formule d'appel, une phrase terminee ou avant "veuillez", interprete-les comme ponctuation et retour ligne.
- Si elles ressemblent a un nom propre plausible ou a un element de dossier, conserve-les et ajoute un warning.

Consignes metier a conserver comme contenu dicte:
- Conserve les instructions de travail: "vous scannez", "vous faites un courriel", "vous joignez", "vous classez", "vous reclassez", "vous enregistrez", "vous invitez", "vous affectez", "vous remettez le dossier a l'archivage", "vous demandez a Yamina", "vous laissez une copie", "vous notez le dossier".
- Ne les execute pas, ne les transforme pas en actions hors texte et ne les supprime pas.
- Elles peuvent coexister avec des commandes de mise en forme dans la meme phrase: conserve la consigne metier, applique seulement la ponctuation et les retours ligne.

Corrections lexicales frequentes autorisees si le contexte est clair:
- "6 juin", "si joint", "s'y joindre" pres de "vous trouverez" ou "je vous communique" -> "ci-joint".
- "outlouc" -> "Outlook".
- "extrait cabis", "extrait cabiste", "extrait qu'abisse" -> "extrait Kbis".
- "procédé verbal" -> "procès-verbal".
- "code jaune" dans un dossier physique -> "cote jaune".
- Ne corrige pas un nom propre ou une reference si l'epellation dictee ne permet pas de trancher.

Exemples de transformation:
- Source: "Mon cher confrere point de sautée ligne je fais suite"
  Sortie: "Mon cher confrere." puis retour a la ligne, puis "Je fais suite"
- Source: "Chere madame Gensotéline je fais suite"
  Sortie: "Chere Madame," puis retour a la ligne, puis "Je fais suite"
- Source: "vous scannez le rapport puis vous faites un courriel"
  Sortie: "Vous scannez le rapport, puis vous faites un courriel"
- Source: "vous trouverez 6 juin l'avis de virement"
  Sortie: "Vous trouverez ci-joint l'avis de virement"
- Source: "il m'a demande de sauter une ligne dans le document"
  Sortie: "Il m'a demande de sauter une ligne dans le document"

Sortie obligatoire:
- Reponds uniquement avec un objet JSON.
- Le JSON doit contenir smart_text et warnings.
- smart_text contient uniquement la retranscription intelligente finale.
- warnings contient seulement les difficultes non resolues, sinon [].
"""


class DirectPostProcessProvider(Protocol):
    def generate_smart_text(self, chunk: str, *, chunk_index: int, total_chunks: int) -> str | dict:
        ...


@dataclass(frozen=True)
class PostProcessProgress:
    stage: str
    detail: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    completed_chunks: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None


PostProcessProgressCallback = Callable[[PostProcessProgress], None]


@dataclass
class DirectPostProcessResult:
    text: str
    warnings: list[str]
    chunks: int


def strip_json_fence(content: str) -> str:
    content = content.strip()
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_smart_payload(content: str | dict) -> tuple[str, list[str]]:
    if isinstance(content, dict):
        payload = content
    else:
        payload = json.loads(strip_json_fence(content))
    if not isinstance(payload, dict):
        raise ValueError("reponse JSON invalide")

    smart_text = payload.get("smart_text")
    if not isinstance(smart_text, str):
        raise ValueError("champ smart_text absent")

    raw_warnings = payload.get("warnings", [])
    warnings = [str(item) for item in raw_warnings] if isinstance(raw_warnings, list) else []
    return smart_text, warnings


class LlamaCppDirectProvider:
    def __init__(self, model_path: Path):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python est absent") from exc

        self.model_path = model_path
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=8192,
            verbose=False,
            chat_format="chatml",
        )

    def generate_smart_text(self, chunk: str, *, chunk_index: int, total_chunks: int) -> str:
        user_prompt = (
            f"Segment {chunk_index + 1}/{total_chunks} de la transcription brute.\n"
            "Transforme ce segment selon les instructions systeme. "
            "Trie explicitement les expressions suspectes entre commandes de mise en forme, "
            "consignes metier a conserver, corrections ASR sures et texte a preserver. "
            "Ne traite pas les consignes metier comme des actions a executer hors texte; "
            "elles doivent rester dans la retranscription si elles sont dictees comme contenu. "
            "N'applique pas de correction risquee sur les noms propres, montants, dates ou references.\n\n"
            "TRANSCRIPTION BRUTE:\n"
            f"{chunk}"
        )
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            top_p=0.1,
            max_tokens=4096,
            response_format={
                "type": "json_object",
                "schema": SMART_OUTPUT_SCHEMA,
            },
        )
        return result["choices"][0]["message"]["content"]


class DirectLLMPostProcessor:
    def __init__(
        self,
        config: AppConfig,
        provider: DirectPostProcessProvider | None = None,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ):
        self.config = config
        self.provider = provider
        self.max_chunk_chars = max_chunk_chars
        self.actions: list[str] = []

    def available(self) -> tuple[bool, str]:
        if self.provider is not None:
            return True, ""
        model_path = self.config.local_postprocess_model_path()
        if not model_path:
            return False, "modele LLM local absent"
        try:
            self.provider = LlamaCppDirectProvider(model_path)
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def apply(
        self,
        raw_text: str,
        *,
        progress_callback: PostProcessProgressCallback | None = None,
        cancel_token=None,
    ) -> DirectPostProcessResult:
        if not raw_text.strip():
            return DirectPostProcessResult(text="", warnings=[], chunks=0)

        started_at = time.monotonic()
        self._emit(
            progress_callback,
            PostProcessProgress(
                stage="loading",
                detail="Chargement du LLM local",
                elapsed_seconds=0.0,
            ),
        )
        ok, reason = self._ensure_available_with_heartbeat(progress_callback, cancel_token, started_at)
        if not ok:
            raise PostProcessUnavailableError(reason)
        assert self.provider is not None

        chunks = self._split_chunks(raw_text)
        self._emit(
            progress_callback,
            PostProcessProgress(
                stage="ready",
                detail=f"{len(chunks)} segment(s) a traiter",
                total_chunks=len(chunks),
                elapsed_seconds=time.monotonic() - started_at,
            ),
        )
        rendered: list[str] = []
        warnings: list[str] = []
        for index, chunk in enumerate(chunks):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            self._emit(
                progress_callback,
                PostProcessProgress(
                    stage="chunk_start",
                    detail=f"Segment {index + 1}/{len(chunks)}",
                    chunk_index=index,
                    total_chunks=len(chunks),
                    completed_chunks=index,
                    elapsed_seconds=time.monotonic() - started_at,
                ),
            )
            try:
                payload = self._generate_chunk_with_heartbeat(
                    chunk,
                    index,
                    len(chunks),
                    progress_callback,
                    cancel_token,
                    started_at,
                )
                smart_text, chunk_warnings = parse_smart_payload(payload)
            except OperationCancelledError:
                raise
            except Exception as exc:
                raise PostProcessUnavailableError(f"reponse LLM invalide ({exc})") from exc

            if not smart_text.strip():
                raise PostProcessUnavailableError("reponse LLM vide")
            rendered.append(smart_text.strip())
            warnings.extend(chunk_warnings)
            completed = index + 1
            elapsed = time.monotonic() - started_at
            eta = self._estimate_remaining(elapsed, completed, len(chunks))
            self._emit(
                progress_callback,
                PostProcessProgress(
                    stage="chunk_done",
                    detail=f"{completed}/{len(chunks)} segment(s) traite(s)",
                    chunk_index=index,
                    total_chunks=len(chunks),
                    completed_chunks=completed,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                ),
            )

        text = self._join_chunks(rendered)
        self.actions.append(f"Post-traitement LLM local : {len(chunks)} segment(s) traite(s).")
        self._emit(
            progress_callback,
            PostProcessProgress(
                stage="done",
                detail="Post-traitement LLM termine",
                total_chunks=len(chunks),
                completed_chunks=len(chunks),
                elapsed_seconds=time.monotonic() - started_at,
                eta_seconds=0,
            ),
        )
        return DirectPostProcessResult(text=text, warnings=warnings, chunks=len(chunks))

    def _ensure_available_with_heartbeat(
        self,
        progress_callback: PostProcessProgressCallback | None,
        cancel_token,
        started_at: float,
    ) -> tuple[bool, str]:
        if self.provider is not None:
            return True, ""
        model_path = self.config.local_postprocess_model_path()
        if not model_path:
            return False, "modele LLM local absent"

        def load_provider() -> tuple[bool, str]:
            return self.available()

        return self._run_blocking_with_heartbeat(
            load_provider,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            started_at=started_at,
            stage="loading",
            detail="Chargement du LLM local",
            chunk_index=0,
            total_chunks=0,
            completed_chunks=0,
        )

    def _generate_chunk_with_heartbeat(
        self,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        progress_callback: PostProcessProgressCallback | None,
        cancel_token,
        started_at: float,
    ) -> str | dict:
        assert self.provider is not None

        def generate() -> str | dict:
            return self.provider.generate_smart_text(
                chunk,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )

        return self._run_blocking_with_heartbeat(
            generate,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            started_at=started_at,
            stage="chunk_running",
            detail=f"Segment {chunk_index + 1}/{total_chunks} en cours",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            completed_chunks=chunk_index,
        )

    def _run_blocking_with_heartbeat(
        self,
        operation: Callable[[], object],
        *,
        progress_callback: PostProcessProgressCallback | None,
        cancel_token,
        started_at: float,
        stage: str,
        detail: str,
        chunk_index: int,
        total_chunks: int,
        completed_chunks: int,
    ):
        results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                results.put((True, operation()))
            except Exception as exc:  # pragma: no cover - exercised through caller paths
                results.put((False, exc))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        cancelled_seen = False
        last_emit = 0.0
        while thread.is_alive():
            elapsed = time.monotonic() - started_at
            if cancel_token is not None and cancel_token.cancelled:
                cancelled_seen = True
            if elapsed - last_emit >= 1.0 or last_emit == 0.0:
                last_emit = elapsed
                heartbeat_detail = detail
                if cancelled_seen:
                    heartbeat_detail = "Annulation demandee ; attente de la fin de l'etape en cours"
                self._emit(
                    progress_callback,
                    PostProcessProgress(
                        stage=stage,
                        detail=heartbeat_detail,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        completed_chunks=completed_chunks,
                        elapsed_seconds=elapsed,
                    ),
                )
            thread.join(timeout=0.2)

        ok, payload = results.get()
        if cancelled_seen and cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if ok:
            return payload
        raise payload

    @staticmethod
    def _emit(progress_callback: PostProcessProgressCallback | None, event: PostProcessProgress) -> None:
        if progress_callback:
            progress_callback(event)

    @staticmethod
    def _estimate_remaining(elapsed_seconds: float, completed: int, total: int) -> float | None:
        if completed <= 0 or total <= completed:
            return 0 if total and total <= completed else None
        average = elapsed_seconds / completed
        return max(0.0, average * (total - completed))

    def _split_chunks(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_chars:
            return [text]

        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if current and len(current) + len(line) > self.max_chunk_chars:
                chunks.append(current)
                current = ""
            if len(line) > self.max_chunk_chars:
                for index in range(0, len(line), self.max_chunk_chars):
                    piece = line[index : index + self.max_chunk_chars]
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(piece)
            else:
                current += line
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _join_chunks(chunks: list[str]) -> str:
        parts = [chunk.strip() for chunk in chunks if chunk.strip()]
        text = "\n\n".join(parts)
        return text + ("\n" if text else "")
