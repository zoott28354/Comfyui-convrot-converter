@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Ignore extra package mirrors/indexes from the user's global pip configuration
rem (for example, an unreachable pypi.ngc.nvidia.com mirror).
set "PIP_CONFIG_FILE=NUL"
set "PIP_EXTRA_INDEX_URL="
set "PIP_TRUSTED_HOST="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

if exist ".venv" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
    if not errorlevel 1 goto :install

    echo The existing .venv is incomplete or points to a Python installation that no longer exists.
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
if defined PYTHON_EXE (
    call :show_selected_python
    choice /C YN /N /M "Use this Python installation? [Y/N]: "
    if not errorlevel 2 goto :create_venv
)

:choose_python
echo.
echo Select the Python installation to use:
echo   1. System default Python
echo   2. Python 3.13 through Python Launcher
echo   3. Python 3.12 through Python Launcher
echo   4. Custom python.exe path
echo   5. Cancel
set "PYTHON_CHOICE="
set /p "PYTHON_CHOICE=Choice [1-5]: "

if "%PYTHON_CHOICE%"=="1" (
    call :find_system_python
    goto :validate_choice
)
if "%PYTHON_CHOICE%"=="2" (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3.13"
    goto :validate_choice
)
if "%PYTHON_CHOICE%"=="3" (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3.12"
    goto :validate_choice
)
if "%PYTHON_CHOICE%"=="4" (
    set "CUSTOM_PYTHON="
    set /p "CUSTOM_PYTHON=Full path to python.exe: "
    set "PYTHON_EXE=!CUSTOM_PYTHON:"=!"
    set "PYTHON_ARGS="
    goto :validate_choice
)
if "%PYTHON_CHOICE%"=="5" exit /b 1
echo Invalid choice.
goto :choose_python

:validate_choice
call :validate_python
if errorlevel 1 (
    echo.
    echo That interpreter is unavailable or unsupported.
    echo Use standard 64-bit CPython 3.12 or 3.13, not a free-threaded build.
    goto :choose_python
)
call :show_selected_python

:create_venv
echo Creating virtual environment...
"%PYTHON_EXE%" %PYTHON_ARGS% -m venv .venv
if errorlevel 1 goto :error

:install
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

:find_system_python
set "PYTHON_EXE="
set "PYTHON_ARGS="
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    call :validate_python
    if not errorlevel 1 exit /b 0
)
set "PYTHON_EXE="
set "PYTHON_ARGS="
where python >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
    set "PYTHON_ARGS="
    call :validate_python
    if not errorlevel 1 exit /b 0
)
set "PYTHON_EXE="
set "PYTHON_ARGS="
exit /b 1

:validate_python
if not defined PYTHON_EXE exit /b 1
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import struct,sys,sysconfig; ok=(3,12) ^<= sys.version_info[:2] ^<= (3,13) and struct.calcsize('P')==8 and not sysconfig.get_config_var('Py_GIL_DISABLED'); raise SystemExit(0 if ok else 1)" >nul 2>nul
exit /b %errorlevel%

:show_selected_python
for /f "delims=" %%V in ('"%PYTHON_EXE%" %PYTHON_ARGS% -c "import platform; print(platform.python_version())"') do set "PYTHON_VERSION=%%V"
for /f "delims=" %%X in ('"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print(sys.executable)"') do set "PYTHON_PATH=%%X"
echo Detected Python %PYTHON_VERSION%
echo   %PYTHON_PATH%
exit /b 0

:error
echo.
echo Installation failed. Review the messages above.
pause
exit /b 1
