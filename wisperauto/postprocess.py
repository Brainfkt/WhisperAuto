"""Rule-based local post-processing for French legal dictation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .commands import load_voice_commands


MODE_RAW = "raw"
MODE_CLEANED = "cleaned"
MODE_SMART = "smart"
MODE_REPORT = "report"

MODE_LABELS = {
    MODE_RAW: "Transcription brute",
    MODE_CLEANED: "Transcription nettoyee",
    MODE_SMART: "Transcription intelligente",
    MODE_REPORT: "Compte rendu structure",
}


FILLER_PATTERN = re.compile(
    r"\b(?:euh+|heu+|hum+|bah|ben|voila|voilà)\b[, ]*",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "zero": 0,
    "zéro": 0,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "vingt": 20,
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
}

MONTHS = {
    "janvier",
    "fevrier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
    "décembre",
}

NUMBER_TOKEN_RE = (
    r"(?:zero|zéro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|"
    r"douze|treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|"
    r"soixante|cent|cents|mille|et)(?:[-\s]+(?:zero|zéro|un|une|deux|trois|"
    r"quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|"
    r"seize|vingt|trente|quarante|cinquante|soixante|cent|cents|mille|et))*"
)


@dataclass
class PostProcessResult:
    outputs: dict[str, str]
    actions: list[str]


def normalize_text(value: str) -> str:
    value = value.lower().replace("’", "'").replace("-", " ")
    decomposed = unicodedata.normalize("NFD", value)
    value = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def parse_french_number(text: str) -> int | None:
    tokens = normalize_text(text).split()
    if not tokens:
        return None

    total = 0
    current = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "et":
            index += 1
            continue
        if token == "quatre" and index + 1 < len(tokens) and tokens[index + 1].startswith("vingt"):
            current += 80
            index += 2
            continue
        if token == "soixante" and index + 1 < len(tokens) and tokens[index + 1] in {
            "dix",
            "onze",
            "douze",
            "treize",
            "quatorze",
            "quinze",
            "seize",
        }:
            current += 60 + NUMBER_WORDS[tokens[index + 1]]
            index += 2
            continue
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
        elif token in {"cent", "cents"}:
            current = max(current, 1) * 100
        elif token == "mille":
            total += max(current, 1) * 1000
            current = 0
        else:
            return None
        index += 1

    return total + current


def remove_last_sentence(text: str) -> str:
    text = text.rstrip()
    match = re.search(r"(.+[.!?]\s+)[^.!?]*$", text, re.DOTALL)
    if match:
        return match.group(1).rstrip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) > 1:
        return " ".join(parts[:-1]).rstrip()
    return ""


def remove_last_word(text: str) -> str:
    return re.sub(r"\s*\S+\s*$", "", text.rstrip())


class PostProcessor:
    def __init__(self, commands_path: Path):
        self.commands_path = commands_path
        self.commands = load_voice_commands(commands_path)
        self.actions: list[str] = []

    def build_outputs(self, raw_text: str) -> PostProcessResult:
        raw = raw_text.strip() + ("\n" if raw_text.strip() else "")
        cleaned = self.process(raw_text, smart=False)
        smart = self.process(raw_text, smart=True)
        report = self.build_report(smart)
        return PostProcessResult(
            outputs={
                MODE_RAW: raw,
                MODE_CLEANED: cleaned,
                MODE_SMART: smart,
                MODE_REPORT: report,
            },
            actions=self.actions,
        )

    def process(self, raw_text: str, smart: bool) -> str:
        output = ""
        list_mode = False
        pending_prefix = ""

        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            line = self._handle_resume_marker(line)
            normalized = normalize_text(line)
            if not normalized:
                continue

            if self._is_exact(normalized, "delete_last_sentence"):
                output = remove_last_sentence(output)
                self.actions.append("Suppression de la derniere phrase.")
                continue
            if self._is_exact(normalized, "delete_last_word"):
                output = remove_last_word(output)
                self.actions.append("Suppression du dernier mot.")
                continue
            if self._is_exact(normalized, "cancel_last"):
                output = remove_last_sentence(output)
                self.actions.append("Annulation du dernier segment.")
                continue
            if self._is_exact(normalized, "restart_sentence"):
                output = remove_last_sentence(output)
                self.actions.append("Reprise de la phrase.")
                continue
            if self._is_exact(normalized, "light_fix"):
                output = self._light_cleanup(output)
                self.actions.append("Nettoyage leger de la phrase precedente.")
                continue

            replacement = self._replacement_command(normalized, output)
            if replacement is not None:
                output = replacement
                continue

            if self._is_exact(normalized, "line_break"):
                output = output.rstrip() + "\n"
                self.actions.append("Retour a la ligne.")
                continue
            if self._is_exact(normalized, "paragraph_break"):
                output = output.rstrip() + "\n\n"
                self.actions.append("Nouveau paragraphe.")
                continue
            if self._is_exact(normalized, "list_end"):
                list_mode = False
                output = output.rstrip() + "\n"
                self.actions.append("Fin de liste.")
                continue
            if smart and self._is_exact(normalized, "list_start"):
                list_mode = True
                if output and not output.endswith("\n"):
                    output += "\n"
                self.actions.append("Debut de liste.")
                continue
            if smart and self._is_exact(normalized, "bullet"):
                list_mode = True
                output = output.rstrip() + "\n"
                self.actions.append("Nouvelle puce.")
                continue
            if smart and self._is_exact(normalized, "title_next"):
                pending_prefix = "# "
                self.actions.append("Titre applique a la phrase suivante.")
                continue
            if smart and self._is_exact(normalized, "subtitle_next"):
                pending_prefix = "## "
                self.actions.append("Sous-titre applique a la phrase suivante.")
                continue

            numbered = self._extract_numbered_item(line)
            if smart and numbered:
                number, item_text = numbered
                if output and not output.endswith("\n"):
                    output += "\n"
                output += f"{number}. {self._prepare_content(item_text, smart).strip()}\n"
                self.actions.append("Element de liste numerotee.")
                continue

            prepared = self._prepare_content(line, smart)
            if not prepared:
                continue

            if smart and list_mode:
                if output and not output.endswith("\n"):
                    output += "\n"
                output += f"- {prepared.strip()}\n"
            elif smart and pending_prefix:
                if output and not output.endswith("\n"):
                    output += "\n"
                output += pending_prefix + prepared.strip().capitalize() + "\n"
                pending_prefix = ""
            else:
                if output and not output.endswith(("\n", " ")):
                    output += " "
                output += prepared.strip()

        return self._final_cleanup(output)

    def build_report(self, smart_text: str) -> str:
        content = smart_text.strip()
        if not content:
            return ""
        lines = [
            "Compte rendu structure",
            "",
            "Note : sortie locale non generative, sans information ajoutee.",
            "",
            "Contenu retranscrit",
            "",
            content,
            "",
        ]
        return "\n".join(lines)

    def _prepare_content(self, line: str, smart: bool) -> str:
        if smart:
            line = self._normalize_dictation_variants(line)
            line = self._apply_structural_markers(line)
            line = self._apply_inline_formatting(line)
            line = self._apply_plural_markers(line)
        line = self._apply_punctuation(line)
        if smart:
            line = self._apply_legal_corrections(line)
        line = self._apply_numeric_rules(line)
        line = FILLER_PATTERN.sub("", line)
        line = self._cleanup_email_like_text(line)
        return self._cleanup_email_like_text(self._fix_spacing(line))

    def _is_exact(self, normalized: str, key: str) -> bool:
        return normalized in {normalize_text(item) for item in self.commands.get(key, [])}

    def _handle_resume_marker(self, line: str) -> str:
        markers = [normalize_text(item) for item in self.commands.get("resume_marker", [])]
        normalized = normalize_text(line)
        for marker in markers:
            if marker in normalized and normalized != marker:
                parts = re.split(marker, normalized, maxsplit=1)
                if len(parts) == 2:
                    self.actions.append("Reprise detectee dans le segment.")
                    return parts[1].strip()
        return "" if normalized in markers else line

    def _replacement_command(self, normalized_line: str, output: str) -> str | None:
        match = re.match(r"remplace\s+(.+?)\s+par\s+(.+)$", normalized_line)
        if match:
            old, new = match.groups()
            if old in normalize_text(output):
                pattern = re.compile(re.escape(old), re.IGNORECASE)
                self.actions.append("Remplacement dicte applique.")
                return pattern.sub(new, output, count=1)
            self.actions.append("Remplacement dicte ignore : texte introuvable.")
            return output

        match = re.match(r"non\s+je voulais dire\s+(.+)$", normalized_line)
        if match:
            self.actions.append("Correction 'je voulais dire' appliquee.")
            base = remove_last_sentence(output)
            if base and not base.endswith((" ", "\n")):
                base += " "
            return base + match.group(1)
        return None

    def _extract_numbered_item(self, line: str) -> tuple[int, str] | None:
        normalized = normalize_text(line)
        for phrase, number in self.commands.get("numbered_list", {}).items():
            normalized_phrase = normalize_text(phrase)
            if normalized == normalized_phrase:
                return number, ""
            if normalized.startswith(normalized_phrase + " "):
                return number, line[len(phrase) :].strip(" ,:-")
        return None

    def _normalize_dictation_variants(self, line: str) -> str:
        line = re.sub(
            r"\b(?:point|pointe)[-\s]+(?:de\s+)?(?:saut(?:er|ez|e|é|ée|ees|ées|es)?|sautes)(?:\s+et)?\s+ligne\b",
            "point sauter ligne",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\b(?:point|pointe)\s+de\s+cette\s+ligne\b",
            "point sauter ligne",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"([.!?])\s*(?:de\s+)?cette\s+ligne\b",
            r"\1 sauter ligne",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\b(?:saut(?:er|ez|e|é|ée|ees|ées|es)?|sautes)(?:\s+et)?\s+ligne\b",
            "sauter ligne",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bouvrez?\s+(?:une\s+)?parenth[eè]se\b",
            "ouvrir parenthèse",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bfermez?\s+(?:la\s+)?parenth[eè]se\b",
            "fermer parenthèse",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"(?<!^)(?<!\n)\b(Dossier\s+)",
            r"\n\n\1",
            line,
        )
        return line

    def _apply_structural_markers(self, line: str) -> str:
        bullet_words = (
            "premier",
            "première",
            "premiere",
            "deuxième",
            "deuxieme",
            "troisième",
            "troisieme",
            "quatrième",
            "quatrieme",
            "cinquième",
            "cinquieme",
            "nouveau",
            "nouvelle",
        )
        line = re.sub(
            rf"(?:^|[\n,.;:]\s*)(?:{'|'.join(bullet_words)})\s*(?:-|tiret|tiré|tire)\s*[,.]?\s*",
            "\n- ",
            line,
            flags=re.IGNORECASE,
        )
        numbered_words = {
            "premièrement": "1.",
            "premierement": "1.",
            "deuxièmement": "2.",
            "deuxiemement": "2.",
            "troisièmement": "3.",
            "troisiemement": "3.",
            "quatrièmement": "4.",
            "quatriemement": "4.",
            "cinquièmement": "5.",
            "cinquiemement": "5.",
        }
        for phrase, replacement in numbered_words.items():
            line = re.sub(
                rf"(?:^|[\n.;:]\s*){phrase}\s*[,.]?\s*",
                f"\n{replacement} ",
                line,
                flags=re.IGNORECASE,
            )
        return line

    def _apply_inline_formatting(self, line: str) -> str:
        line = re.sub(
            r"\bentre guillemets\s+(.+)$",
            lambda match: f"« {match.group(1).strip()} »",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\ben gras\s+(.+)$",
            lambda match: f"**{match.group(1).strip()}**",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\ben italique\s+(.+)$",
            lambda match: f"*{match.group(1).strip()}*",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bsouligne\s+(.+)$",
            lambda match: f"_{match.group(1).strip()}_",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bmets en majuscule\s+(.+)$",
            lambda match: match.group(1).upper(),
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\ben majuscule\s+([A-Za-z0-9][A-Za-z0-9.-]{1,20})",
            lambda match: match.group(1).upper(),
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bmajuscule a\s+(\w+)",
            lambda match: match.group(1).capitalize(),
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bmajuscule à\s+(\w+)",
            lambda match: match.group(1).capitalize(),
            line,
            flags=re.IGNORECASE,
        )
        return line

    def _apply_plural_markers(self, line: str) -> str:
        return re.sub(
            r"\b([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]{1,})\s*,?\s+au\s+pluriel\b",
            lambda match: self._pluralize(match.group(1)),
            line,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _pluralize(word: str) -> str:
        if word.lower().endswith(("s", "x")):
            return word
        return word + "s"

    def _apply_legal_corrections(self, line: str) -> str:
        corrections = [
            (r"\bcourt d'appel\b", "cour d'appel"),
            (r"\bCour d'Etat\b", "Conseil d'Etat"),
            (r"\bConseil d'Etat\b", "Conseil d'Etat"),
            (r"\bconseil de prud'homme\b", "conseil de prud'hommes"),
            (r"\bConseil de prud'homme\b", "Conseil de prud'hommes"),
            (r"\bproc[eé]d[eé] verbal\b", "procès-verbal"),
            (r"\bproces verbal\b", "procès-verbal"),
            (r"\bcompte Carpa\b", "compte CARPA"),
            (r"\bCarpa\b", "CARPA"),
            (r"\bextrait qu[' ]?abisse\b", "extrait Kbis"),
            (r"\bextrait qu[' ]?habit\b", "extrait Kbis"),
            (r"\bextrait kbis\b", "extrait Kbis"),
            (r"\bsi joint\b", "ci-joint"),
            (r"\bs'y joint\b", "ci-joint"),
            (r"\bs'y joindre\b", "ci-joint"),
            (r"\bsi jointe\b", "ci-jointe"),
            (r"\bs'y jointe\b", "ci-jointe"),
        ]
        for pattern, replacement in corrections:
            line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)

        line = re.sub(
            r"\b(vous trouverez|je vous communique|vous trouverez ci-dessous)\s+6\s+juin\b",
            lambda match: f"{match.group(1)} ci-joint",
            line,
            flags=re.IGNORECASE,
        )
        return line

    def _apply_punctuation(self, line: str) -> str:
        replacements = self.commands.get("punctuation", {})
        for phrase in sorted(replacements, key=len, reverse=True):
            value = replacements[phrase]
            pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
            line = pattern.sub(lambda match: self._replace_punctuation_if_context_allows(line, match, value), line)

        line = self._replace_command_phrases(line, "paragraph_break", "\n\n")
        line = self._replace_command_phrases(line, "line_break", "\n")
        line = self._replace_command_phrases(line, "double_space", "  ")
        line = self._replace_command_phrases(line, "space", " ")
        return line

    def _replace_command_phrases(self, line: str, key: str, replacement: str) -> str:
        for phrase in sorted(self.commands.get(key, []), key=len, reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
            line = pattern.sub(lambda match: self._replace_if_context_allows(line, match, replacement), line)
        return line

    def _replace_punctuation_if_context_allows(
        self,
        line: str,
        match: re.Match,
        replacement: str,
    ) -> str:
        before = normalize_text(line[max(0, match.start() - 28) : match.start()])
        after = normalize_text(line[match.end() : match.end() + 36])
        punctuation_keep_markers = [
            "le mot",
            "les mots",
            "l'expression",
            "l expression",
            "terme",
            "ecrire",
            "dire",
            "prononcer",
        ]
        if any(marker in before for marker in punctuation_keep_markers):
            return match.group(0)
        if normalize_text(match.group(0)) == "pointe":
            if after and not after.startswith(("sauter ligne", "a la ligne", "nouvelle ligne")):
                return match.group(0)
        self.actions.append(f"Commande vocale appliquee : {match.group(0)}")
        return replacement

    def _replace_if_context_allows(self, line: str, match: re.Match, replacement: str) -> str:
        before = normalize_text(line[max(0, match.start() - 50) : match.start()])
        after = normalize_text(line[match.end() : match.end() + 50])
        markers = [normalize_text(item) for item in self.commands.get("context_keep_markers", [])]
        if any(marker in before or marker in after for marker in markers):
            return match.group(0)
        self.actions.append(f"Commande vocale appliquee : {match.group(0)}")
        return replacement

    def _apply_numeric_rules(self, line: str) -> str:
        line = re.sub(
            rf"\barticle\s+({NUMBER_TOKEN_RE})\b",
            lambda match: self._replace_number_context(match, "article {number}"),
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            rf"\b({NUMBER_TOKEN_RE})\s+euros?\b",
            lambda match: self._replace_number_context(match, "{number} €"),
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            rf"\b({NUMBER_TOKEN_RE})\s+pourcentage\b",
            lambda match: self._replace_number_context(match, "{number} %"),
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            rf"\ble\s+({NUMBER_TOKEN_RE})\s+({'|'.join(MONTHS)})\s+({NUMBER_TOKEN_RE})\b",
            self._replace_date,
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            rf"\bnumero de dossier\s+((?:[a-z]\s+){{2,}})({NUMBER_TOKEN_RE})\b",
            self._replace_case_number,
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            rf"\bnuméro de dossier\s+((?:[a-z]\s+){{2,}})({NUMBER_TOKEN_RE})\b",
            self._replace_case_number,
            line,
            flags=re.IGNORECASE,
        )
        return line

    def _replace_number_context(self, match: re.Match, template: str) -> str:
        number = parse_french_number(match.group(1))
        if number is None:
            return match.group(0)
        self.actions.append("Nombre dicte normalise.")
        return template.format(number=number)

    def _replace_date(self, match: re.Match) -> str:
        day = parse_french_number(match.group(1))
        year = parse_french_number(match.group(3))
        if day is None or year is None:
            return match.group(0)
        self.actions.append("Date dictee normalisee.")
        return f"le {day} {match.group(2)} {year}"

    def _replace_case_number(self, match: re.Match) -> str:
        letters = "".join(match.group(1).split()).upper()
        number = parse_french_number(match.group(2))
        if not letters or number is None:
            return match.group(0)
        self.actions.append("Numero de dossier normalise.")
        return f"numero de dossier {letters}-{number}"

    @staticmethod
    def _light_cleanup(text: str) -> str:
        text = FILLER_PATTERN.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _fix_spacing(text: str) -> str:
        text = re.sub(r"\(\s*,\s*", "(", text)
        text = re.sub(r",\s*\)", ")", text)
        text = re.sub(r"([.!?])\s*(?:de\s+)?sauter ligne", r"\1\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*,\s*([.!?])", r"\1", text)
        text = re.sub(r"([.!?])\s*,+", r"\1", text)
        text = re.sub(r"(?:,\s*){2,}", ", ", text)
        text = re.sub(r"\.\s+\.", ".", text)
        text = re.sub(r"\s+([,.;:?!])", r"\1", text)
        text = re.sub(r"([,.;:?!])(?=[^\s])", r"\1 ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+»", " »", text)
        text = re.sub(r"«\s+", "« ", text)
        text = re.sub(r"[ \t]{3,}", "  ", text)
        return text.strip()

    @staticmethod
    def _cleanup_email_like_text(text: str) -> str:
        text = re.sub(r"\s*@\s*", "@", text)
        text = re.sub(r"(?<=@)\s+", "", text)
        text = re.sub(r"(?<=\w)\s*\.\s*(?=(?:fr|com|net|org|eu)\b)", ".", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\s+\.\s+([A-Za-z]{2,})\b", r"\1@\2.\3", text)
        return text

    def _final_cleanup(self, text: str) -> str:
        text = self._fix_spacing(text)
        text = self._cleanup_email_like_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + ("\n" if text.strip() else "")
