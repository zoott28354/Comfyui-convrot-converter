@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Python environment not found. Starting setup.bat...
    call setup.bat
    if errorlevel 1 exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0convrot_gui.py"
