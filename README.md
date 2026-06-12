# WisperAuto

WisperAuto est une application locale de transcription audio pour un usage juridique. Elle est pensée pour rester légère, simple à installer, et utilisable sur un poste client sans envoyer les audios vers un service externe.

## Ce que fait l'application

- importe un ou plusieurs fichiers audio ;
- transcrit localement avec un moteur Whisper ;
- conserve la transcription brute ;
- produit une transcription intelligente avec un LLM local ;
- permet de copier, exporter ou supprimer les transcriptions ;
- garde un historique local.

Modes disponibles :

- **Transcription brute** : sortie directe du moteur audio.
- **Transcription intelligente** : post-traitement LLM local qui interprète la ponctuation, les retours à la ligne, les listes et les commandes vocales évidentes.
- **Compte rendu structuré** : sortie organisée à partir de la transcription intelligente, sans remplacer l'original.

## Confidentialité

Les audios restent sur l'ordinateur. WisperAuto ne les envoie pas en ligne.

Les seules actions qui utilisent Internet sont explicites et lancées depuis l'interface :

- installation de packages ;
- téléchargement de modèles Whisper ;
- téléchargement du modèle LLM local ;
- ouverture de la page Hugging Face pour créer un token.

Évitez de placer le dossier de données WisperAuto dans un dossier synchronisé cloud si les audios ou transcriptions sont sensibles.

## Installation rapide Windows

1. Installer Python **3.11 ou 3.12** depuis :
   <https://www.python.org/downloads/windows/>

2. Décompresser ou cloner ce dossier sur l'ordinateur, par exemple :

   ```text
   C:\WisperAuto
   ```

3. Double-cliquer :

   ```bat
   start_whisper.bat
   ```

Le script crée automatiquement un environnement local `.venv` si nécessaire. Il évite Python Anaconda global et Python 3.13 pour le déploiement client.

Au premier lancement, ouvrir **Paramètres > Téléchargements**, puis installer dans cet ordre :

1. **FFmpeg / ffprobe**.
2. **Moteur faster-whisper** ou laisser le moteur sur **Auto**.
3. **Modèle moteur**.
4. **Post-traitement LLM local**.
5. **Modèle post-traitement LLM GGUF**.

Le modèle recommandé pour une bonne UX locale est `large-v3-turbo`. Si le poste est lent ou sans GPU, commencer par `medium`.

## Installation macOS / Linux

Depuis le dossier du projet :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python auto_transcribe.py
```

Python 3.11 fonctionne aussi.

Ensuite, installer les dépendances depuis **Paramètres > Téléchargements** comme sur Windows.

Notes :

- Sur macOS, FFmpeg peut être installé depuis l'interface si Homebrew est présent.
- Sur Mac Apple Silicon, le moteur optionnel **MLX Mac** peut être plus rapide s'il est installé.
- Sur Linux, si Tkinter est absent, installez le paquet système Python Tk correspondant à votre distribution.

## Premier usage

1. Cliquer **Ajouter**.
2. Sélectionner un ou plusieurs fichiers `.mp3`, `.wav`, `.m4a`, `.dss` ou `.ds2`.
3. Cliquer **Transcrire** ou **Tout**.
4. Suivre la progression en bas de l'application.
5. Pendant le post-traitement intelligent, l'interface affiche le chargement LLM, le segment en cours, le temps écoulé et une ETA quand elle est fiable.
6. Choisir le mode de sortie : brute, intelligente ou compte rendu.
7. Utiliser **Copier**, **Exporter** ou **Dossier**.

Le bouton **Annuler** stoppe un traitement actif dès que possible. Pendant un appel LLM local long, l'annulation est prise en compte à la fin du segment en cours.

Le bouton **Supprimer** retire l'entrée d'historique et les fichiers `.txt` générés. Les audios archivés ou en erreur sont conservés pour éviter une perte.

## Moteurs disponibles

- **Auto** : choisit le meilleur moteur local disponible.
- **faster-whisper** : moteur universel recommandé, fiable sur Windows, macOS et Linux.
- **MLX Mac** : option Apple Silicon.
- **whisper.cpp** : option CPU/Metal portable avec modèle quantized.

Profils :

- **Rapide** : meilleure UX sur poste standard.
- **Équilibré** : compromis vitesse/précision.
- **Précis** : plus lent, utile si la qualité prime.

Le bouton **Tester performances** compare les moteurs installés sur un extrait local et recommande le plus rapide pour la machine.

## Hugging Face et téléchargements rapides

Si les téléchargements Hugging Face sont lents :

1. Ouvrir **Paramètres > Téléchargements**.
2. Cliquer **Créer token** pour ouvrir la page officielle Hugging Face.
3. Créer un token en lecture seule.
4. Coller le token dans **HF token**, puis cliquer **Enregistrer**.
5. Activer **Téléchargement rapide Xet** si souhaité.

Le token est stocké localement dans :

```text
<WISPERAUTO_HOME>/settings.json
```

Il n'est pas écrit dans les commandes affichées dans le journal.

## Dossier de données

Par défaut :

```text
Windows : %USERPROFILE%\Documents\WisperAuto
macOS/Linux : ~/Documents/WisperAuto
```

Contenu principal :

- `inbox/` : fichiers ajoutés avant traitement ;
- `processed/` : audios traités ;
- `failed/` : audios en erreur ;
- `outbox/` : transcriptions `.txt` ;
- `logs/history.jsonl` : historique local ;
- `models/` : modèles téléchargés ;
- `settings.json` : réglages utilisateur.

Pour utiliser un autre dossier :

```bash
export WISPERAUTO_HOME="/chemin/vers/WisperAuto"
python auto_transcribe.py
```

Sous Windows :

```bat
set WISPERAUTO_HOME=D:\WisperAutoData
start_whisper.bat
```

## Mode console

Interface graphique :

```bash
python auto_transcribe.py
```

Transcrire un fichier :

```bash
python auto_transcribe.py --once chemin/vers/audio.mp3
```

Surveiller le dossier `inbox` :

```bash
python auto_transcribe.py --watch
```

Autoriser le téléchargement initial d'un modèle en console :

```bash
python auto_transcribe.py --once audio.mp3 --allow-model-download
```

## Dépannage

**L'application démarre mais le modèle est absent**

Ouvrir **Paramètres > Téléchargements**, puis cliquer **Télécharger modèle moteur**.

**Le post-traitement intelligent est indisponible**

Installer :

1. **Post-traitement LLM local** ;
2. **Modèle post-traitement LLM GGUF**.

La transcription brute reste disponible même si le LLM échoue.

**FFmpeg est introuvable**

Installer FFmpeg depuis l'interface. Si le PATH change, fermer puis relancer WisperAuto.

**Téléchargement très lent**

- enregistrer un token Hugging Face ;
- activer le mode Xet rapide ;
- réduire temporairement le modèle (`medium` au lieu de `large-v3` ou `large-v3-turbo`) ;
- vérifier que le poste ne bloque pas Hugging Face via proxy, antivirus ou réseau d'entreprise.

**Transcription lente**

- utiliser le profil **Rapide** ;
- essayer `medium` ;
- lancer **Tester performances** ;
- sur Mac Apple Silicon, tester **MLX Mac** ;
- sur Windows/Linux avec GPU NVIDIA, utiliser **Auto** ou **faster-whisper** avec CUDA disponible.

**DSS/DS2 illisible**

Certains fichiers DSS Pro ou DS2 ne sont pas décodés par FFmpeg. Dans ce cas, convertir avec le logiciel du dictaphone avant import.

## Variables utiles

- `WISPERAUTO_HOME` : dossier de données local.
- `WISPERAUTO_BACKEND` : `auto`, `faster-whisper`, `mlx-whisper`, `whisper.cpp`.
- `WISPERAUTO_MODEL` : `small`, `medium`, `large-v3-turbo`, `large-v3`.
- `WISPERAUTO_OUTPUT_MODE` : `raw`, `smart`, `report`.
- `WISPERAUTO_TRANSCRIPTION_PROFILE` : `fast`, `balanced`, `precise`.
- `WISPERAUTO_HF_TOKEN` ou `HF_TOKEN` : token Hugging Face.
- `WISPERAUTO_HF_FAST_DOWNLOAD=1` : active le mode Hugging Face rapide.
- `WISPERAUTO_MODEL_DOWNLOAD_TIMEOUT_MINUTES` : délai maximal de téléchargement.
