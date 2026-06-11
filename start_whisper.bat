@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if "%WISPERAUTO_HOME%"=="" set "WISPERAUTO_HOME=%USERPROFILE%\Documents\WisperAuto"

if not exist ".venv\Scripts\python.exe" (
    set "WISPERAUTO_PY="
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "WISPERAUTO_PY=py -3.11"
    if "!WISPERAUTO_PY!"=="" (
        py -3.12 -c "import sys" >nul 2>nul
        if not errorlevel 1 set "WISPERAUTO_PY=py -3.12"
    )
    if "!WISPERAUTO_PY!"=="" (
        echo Python 3.11 ou 3.12 est requis pour le venv client.
        echo Installez Python depuis https://www.python.org/downloads/windows/
        pause
        exit /b 1
    )
    !WISPERAUTO_PY! -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)"
    if errorlevel 1 (
        echo Python 3.11 ou 3.12 est requis.
        echo Installez Python depuis https://www.python.org/downloads/windows/
        pause
        exit /b 1
    )
    echo Creation de l'environnement Python local...
    !WISPERAUTO_PY! -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)"
if errorlevel 1 (
    echo L'environnement .venv utilise une version Python non ciblee.
    echo Supprimez le dossier .venv puis relancez start_whisper.bat avec Python 3.11 ou 3.12 installe.
    pause
    exit /b 1
)
python auto_transcribe.py
