@echo off
setlocal
cd /d "%~dp0"

rem Ignore extra package mirrors/indexes from the user's global pip configuration
rem (for example, an unreachable pypi.ngc.nvidia.com mirror).
set "PIP_CONFIG_FILE=NUL"
set "PIP_EXTRA_INDEX_URL="
set "PIP_TRUSTED_HOST="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.12 from python.org and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :error
)

echo Updating pip...
".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple --upgrade pip
if errorlevel 1 goto :error

echo Installing PyTorch with CUDA 12.8 support...
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error

echo Installing interface and ConvRot dependencies...
".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Installation complete. Start the application with START.bat
pause
exit /b 0

:error
echo.
echo Installation failed. Review the messages above.
pause
exit /b 1
