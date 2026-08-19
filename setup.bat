@echo off
rem SPDX-License-Identifier: GPL-3.0-only
rem Copyright (C) 2026 zoott28354 and contributors

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Ignore extra package mirrors/indexes from the user's global pip configuration
rem (for example, an unreachable pypi.ngc.nvidia.com mirror).
set "PIP_CONFIG_FILE=NUL"
set "PIP_EXTRA_INDEX_URL="
set "PIP_TRUSTED_HOST="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

if exist ".venv" (
    ".venv\Scripts\python.exe" -c "import struct,sys,sysconfig; ok=(3,12) <= sys.version_info[:2] <= (3,14) and struct.calcsize('P')==8 and not sysconfig.get_config_var('Py_GIL_DISABLED'); raise SystemExit(0 if ok else 1)" >nul 2>nul
    if not errorlevel 1 goto :install

    echo The existing .venv is incomplete, incompatible, or points to a Python installation that no longer exists.
    choice /C RC /N /M "Press R to recreate it or C to cancel: "
    if errorlevel 2 exit /b 1
    rmdir /s /q ".venv"
    if exist ".venv" (
        echo Unable to remove the old .venv. Close programs using it and try again.
        pause
        exit /b 1
    )
)

call :find_system_python
if not defined PYTHON_EXE goto :python_required
call :show_selected_python

:create_venv
echo Creating virtual environment...
"%PYTHON_EXE%" %PYTHON_ARGS% -m venv .venv
if errorlevel 1 goto :error

:install
echo Updating pip...
".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple --only-binary=:all: pip==26.2.1
if errorlevel 1 goto :error

echo Installing PyTorch with CUDA 12.8 support...
".venv\Scripts\python.exe" -m pip install --only-binary=:all: torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error

echo Installing verified interface and ConvRot dependencies...
".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple --require-hashes -r requirements.lock
if errorlevel 1 goto :error

call :write_launcher
if errorlevel 1 goto :error

echo.
echo Installation complete. Start the application with start.bat
pause
exit /b 0

:find_system_python
set "PYTHON_EXE="
set "PYTHON_ARGS="
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.14 3.13 3.12) do (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-%%V"
        call :validate_python
        if not errorlevel 1 exit /b 0
    )
)
set "PYTHON_EXE="
set "PYTHON_ARGS="
where python >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        set "PYTHON_EXE=%%P"
        set "PYTHON_ARGS="
        call :validate_python
        if not errorlevel 1 exit /b 0
    )
)
set "PYTHON_EXE="
set "PYTHON_ARGS="
exit /b 1

:validate_python
if not defined PYTHON_EXE exit /b 1
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import struct,sys,sysconfig; ok=(3,12) <= sys.version_info[:2] <= (3,14) and struct.calcsize('P')==8 and not sysconfig.get_config_var('Py_GIL_DISABLED'); raise SystemExit(0 if ok else 1)" >nul 2>nul
exit /b %errorlevel%

:show_selected_python
for /f "delims=" %%V in ('"%PYTHON_EXE%" %PYTHON_ARGS% -c "import platform; print(platform.python_version())"') do set "PYTHON_VERSION=%%V"
for /f "delims=" %%X in ('"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print(sys.executable)"') do set "PYTHON_PATH=%%X"
echo Detected Python %PYTHON_VERSION%
echo   %PYTHON_PATH%
exit /b 0

:python_required
echo.
echo Compatible Python was not found.
echo Install standard 64-bit Python 3.12, 3.13, or 3.14, then run setup.bat again:
echo https://www.python.org/downloads/windows/
pause
exit /b 1

:write_launcher
echo Creating start.bat...
>start.bat echo @echo off
>>start.bat echo setlocal
>>start.bat echo cd /d "%%~dp0"
>>start.bat echo if not exist ".venv\Scripts\pythonw.exe" ^(
>>start.bat echo     echo Python environment not found. Run setup.bat first.
>>start.bat echo     pause
>>start.bat echo     exit /b 1
>>start.bat echo ^)
>>start.bat echo start "" ".venv\Scripts\pythonw.exe" "%%~dp0convrot_gui.py"
if not exist "start.bat" exit /b 1
exit /b 0

:error
echo.
echo Installation failed. Review the messages above.
pause
exit /b 1
