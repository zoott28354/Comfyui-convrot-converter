@echo off
setlocal
cd /d "%~dp0"

rem Non usare eventuali mirror/indici extra presenti nella configurazione pip
rem dell'utente (per esempio pypi.ngc.nvidia.com non raggiungibile).
set "PIP_CONFIG_FILE=NUL"
set "PIP_EXTRA_INDEX_URL="
set "PIP_TRUSTED_HOST="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher non trovato. Installa Python 3.12 da python.org e riprova.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creazione ambiente virtuale Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :error
)

echo Aggiornamento pip...
".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple --upgrade pip
if errorlevel 1 goto :error

echo Installazione PyTorch con supporto CUDA 12.8...
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error

echo Installazione dipendenze dell'interfaccia e ConvRot...
".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Installazione completata. Avvia il programma con AVVIA.bat
pause
exit /b 0

:error
echo.
echo Installazione non riuscita. Controlla i messaggi qui sopra.
pause
exit /b 1
