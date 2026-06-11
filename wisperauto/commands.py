"""Configurable voice command dictionary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_COMMANDS = {
    "line_break": [
        "sauter ligne",
        "sauter une ligne",
        "saute ligne",
        "sautez ligne",
        "sauté ligne",
        "sautée ligne",
        "sautes ligne",
        "sautes et ligne",
        "pointe saute ligne",
        "pointe sautée ligne",
        "point saute ligne",
        "point sautée ligne",
        "a la ligne",
        "à la ligne",
        "nouvelle ligne",
    ],
    "paragraph_break": ["nouveau paragraphe", "paragraphe suivant"],
    "space": ["espace"],
    "double_space": ["double espace"],
    "list_start": ["liste", "fais une liste"],
    "bullet": ["nouvelle puce"],
    "list_end": ["fin de liste"],
    "title_next": ["titre", "mets ça en titre", "mets ca en titre"],
    "subtitle_next": ["sous-titre", "sous titre"],
    "delete_last_sentence": ["supprime la dernière phrase", "supprime la derniere phrase"],
    "delete_last_word": ["supprime le dernier mot"],
    "cancel_last": ["annule ça", "annule ca"],
    "restart_sentence": ["recommence la phrase"],
    "resume_marker": ["je reprends"],
    "light_fix": ["corrige ça", "corrige ca"],
    "punctuation": {
        "points de suspension": "...",
        "point d'interrogation": "?",
        "point d’interrogation": "?",
        "point d'exclamation": "!",
        "point d’exclamation": "!",
        "point-virgule": ";",
        "point virgule": ";",
        "deux-points": ":",
        "deux points": ":",
        "virgule": ",",
        "pointe": ".",
        "point": ".",
        "ouvrir les guillemets": "«",
        "fermer les guillemets": "»",
        "ouvrez une parenthèse": "(",
        "ouvrez une parenthese": "(",
        "ouvrez parenthèse": "(",
        "ouvrez parenthese": "(",
        "ouvrir parenthèse": "(",
        "ouvrir parenthese": "(",
        "fermez la parenthèse": ")",
        "fermez la parenthese": ")",
        "fermez parenthèse": ")",
        "fermez parenthese": ")",
        "fermer parenthèse": ")",
        "fermer parenthese": ")",
        "premier tiret": "-",
        "premier tiré": "-",
        "premier tire": "-",
        "deuxième tiret": "-",
        "deuxieme tiret": "-",
        "deuxième tiré": "-",
        "deuxieme tiré": "-",
        "deuxième tire": "-",
        "deuxieme tire": "-",
        "nouveau tiret": "-",
        "nouveau tiré": "-",
        "nouveau tire": "-",
        "tiret": "-",
        "slash": "/",
        "arobase": "@",
    },
    "numbered_list": {
        "numero un": 1,
        "numéro un": 1,
        "numero deux": 2,
        "numéro deux": 2,
        "numero trois": 3,
        "numéro trois": 3,
        "numero quatre": 4,
        "numéro quatre": 4,
        "numero cinq": 5,
        "numéro cinq": 5,
        "numero six": 6,
        "numéro six": 6,
        "numero sept": 7,
        "numéro sept": 7,
        "numero huit": 8,
        "numéro huit": 8,
        "numero neuf": 9,
        "numéro neuf": 9,
        "numero dix": 10,
        "numéro dix": 10,
    },
    "context_keep_markers": [
        "le mot",
        "les mots",
        "l'expression",
        "l’expression",
        "terme",
        "dire",
        "ecrire",
        "écrire",
        "prononcer",
        "a demandé de",
        "a demande de",
        "dans le document",
        "dans le contrat",
    ],
}


def load_voice_commands(path: Path) -> dict:
    """Load user commands, creating a local editable file on first run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(DEFAULT_COMMANDS, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return deepcopy(DEFAULT_COMMANDS)

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    commands = deepcopy(DEFAULT_COMMANDS)
    for key, value in loaded.items():
        existing = commands.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            existing.update(value)
        elif isinstance(value, list) and isinstance(existing, list):
            merged = list(existing)
            for item in value:
                if item not in merged:
                    merged.append(item)
            commands[key] = merged
        else:
            commands[key] = value
    return commands
