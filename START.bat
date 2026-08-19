@echo off
setlocal
cd /d "%~dp0"
set "NEED_SETUP="
if not exist ".venv\Scripts\pythonw.exe" set "NEED_SETUP=1"
if not defined NEED_SETUP (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 set "NEED_SETUP=1"
)
if defined NEED_SETUP (
    echo Python environment not found or no longer valid. Starting setup.bat...
    call setup.bat
    if errorlevel 1 exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0convrot_gui.py"
