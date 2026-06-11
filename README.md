# WisperAuto

Application locale et légère de transcription audio pour usage juridique.

WisperAuto garde deux étapes séparées :

- transcription audio brute avec un moteur local (`Auto`, `faster-whisper`, `MLX Mac` ou `whisper.cpp`) ;
- post-traitement local et configurable pour produire des versions nettoyées ou intelligentes.

## Modes de sortie

- **Transcription brute** : conserve les mots reconnus, y compris les commandes vocales.
- **Transcription nettoyée** : applique ponctuation et paragraphes évidents, supprime quelques hésitations inutiles.
- **Transcription intelligente** : interprète les commandes de mise en forme, listes, titres, corrections simples et références fréquentes.
- **Compte rendu structuré** : organise la sortie sans inventer d'information et sans remplacer la transcription originale.

## Confidentialité

Les fichiers audio restent locaux. Le projet n’envoie pas les audios vers un service externe. Si le modèle Whisper n’est pas présent en local, l’application demande une autorisation explicite avant le téléchargement initial du modèle.

## Lancement

Windows :

```bat
start_whisper.bat
```

Python 3.11 ou 3.12 est recommandé pour le venv local client. Le lanceur Windows
cree le venv local si besoin et evite Python global/Conda/Python 3.13 pour le
deploiement client.

Les installations de `ffmpeg`, du moteur de transcription et du modele peuvent
ensuite etre lancees depuis l'interface dans `Parametres et installation`.
Le modele du moteur choisi doit etre installe avant de lancer une transcription ;
le telechargement affiche un journal avec taille visible, debit et ETA quand
c'est possible. Les installations et telechargements peuvent etre annules.
Depuis les paramètres, il est possible de choisir `small`, `medium`,
`large-v3-turbo` ou `large-v3` avant de télécharger le modèle.

Moteurs disponibles :

- **Auto** : choisit le meilleur moteur local disponible.
- **faster-whisper** : backend universel fiable, CPU `int8` par defaut, CUDA si disponible.
- **MLX Mac** : option Apple Silicon si `mlx-whisper` est installe.
- **whisper.cpp** : option portable via binaire local `whisper-cli` et modele quantized.

Chaque moteur utilise son propre format de modele dans `models/<moteur>/<modele>`.

Profils performance :

- **Rapide** : beam size reduit, batch interne quand possible, VAD plus agressif.
- **Equilibre** : compromis vitesse/precision pour les dictées longues.
- **Precis** : beam size plus eleve, contexte conserve, batch plus prudent.

Dans `Parametres`, le bouton `Tester performances` lance un benchmark local sur
un extrait temporaire du fichier choisi. Il mesure le chargement du modele, le
temps de transcription, le ratio temps reel et recommande le moteur le plus
rapide parmi ceux qui ont deja un modele local.

Flux principal :

1. Cliquer `Ajouter fichiers`.
2. Verifier la file locale.
3. Lancer `Transcrire selection` ou `Tout transcrire`.
4. Suivre la progression par fichier.
5. Copier, exporter ou supprimer l'entree d'historique.

La suppression retire l'entree d'historique et les fichiers `.txt` generes.
Les audios archives ou echoues sont conserves pour eviter une perte.

Console :

```bash
python auto_transcribe.py
python auto_transcribe.py --watch
python auto_transcribe.py --once chemin/vers/audio.mp3
```

## Configuration

Variables utiles :

- `WISPERAUTO_HOME` : dossier de données local.
- `WISPERAUTO_BACKEND` : `auto`, `faster-whisper`, `mlx-whisper` ou `whisper.cpp`.
- `WISPERAUTO_DEVICE` : `auto`, `cpu` ou `cuda` pour `faster-whisper`.
- `WISPERAUTO_MODEL` : modèle, par défaut `large-v3-turbo`.
- `WISPERAUTO_MODEL_PATH` : chemin vers un modèle local déjà téléchargé.
- `WISPERAUTO_MLX_MODEL_PATH` : chemin vers un modèle MLX local.
- `WISPERAUTO_WHISPER_CPP_MODEL_PATH` : chemin vers un modèle `whisper.cpp`.
- `WISPERAUTO_WHISPER_CPP_BINARY` : chemin vers `whisper-cli` si non présent dans le PATH.
- `WISPERAUTO_OUTPUT_MODE` : `raw`, `cleaned`, `smart` ou `report`.
- `WISPERAUTO_TRANSCRIPTION_PROFILE` : `fast`, `balanced` ou `precise`.
- `WISPERAUTO_CPU_THREADS` : threads CPU pour `faster-whisper`, `0` = auto.
- `WISPERAUTO_NUM_WORKERS` : workers CTranslate2, par defaut `1`.
- `WISPERAUTO_BATCH_SIZE` : batch interne `faster-whisper`, `0` = auto par profil.
- `WISPERAUTO_WHISPER_CPP_THREADS` : threads `whisper.cpp`, `0` = auto.
- `WISPERAUTO_WHISPER_CPP_BEAM_SIZE` : beam size `whisper.cpp`, `0` = profil.
- `WISPERAUTO_WHISPER_CPP_BEST_OF` : best-of `whisper.cpp`, `0` = profil.
- `WISPERAUTO_VAD_SILENCE_MS` : silence VAD minimal, `0` = auto par profil.
- `WISPERAUTO_BENCHMARK_SECONDS` : durée de l'extrait benchmark, par defaut `90`.
- `WISPERAUTO_MAX_FILE_MB` : limite de taille fichier.
- `WISPERAUTO_MAX_DURATION_MINUTES` : limite de durée.
- `WISPERAUTO_MODEL_DOWNLOAD_TIMEOUT_MINUTES` : delai maximal de telechargement du modele.
- `WISPERAUTO_DISABLE_HF_XET` : `1` par defaut pour un telechargement plus lisible ; mettre `0` pour reactiver Xet.

Le dictionnaire de commandes vocales est créé dans :

```text
<WISPERAUTO_HOME>/voice_commands.json
```

Un exemple versionné est disponible dans `voice_commands.example.json`.
