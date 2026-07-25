@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
title UMU Quiz Helper - Auto Answer

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "REQ=%BACKEND%\requirements.txt"
set "HOST=0.0.0.0"
set "PORT=8000"
set "OPEN_URL=http://localhost:%PORT%"

echo.
echo ============================================================
echo   UMU Quiz Helper v2.1.0
echo   Auto extract questions + AI auto answer
echo ============================================================
echo   Main page : http://localhost:8000
echo   API docs  : http://localhost:8000/api/docs
echo ============================================================
echo.

if not exist "%BACKEND%\server.py" (
    echo [ERROR] Cannot find backend\server.py.
    echo [ERROR] Please run this script from the project root folder.
    pause
    exit /b 1
)

call :FindPython
if errorlevel 1 exit /b 1

echo [INFO] Python version:
%PYTHON% --version
echo.

echo [INFO] Checking dependencies...
%PYTHON% -m pip install -r "%REQ%" -q
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo [TIP] Try running this manually:
    echo       %PYTHON% -m pip install -r "%REQ%"
    pause
    exit /b 1
)
echo [INFO] Dependencies are ready.
echo.

call :PickPort
set "OPEN_URL=http://localhost:%PORT%"

echo [INFO] Starting server on %OPEN_URL%
echo [INFO] Workflow: Batch tab - paste text - extract questions - generate answers.
echo [INFO] Keep this window open. Press Ctrl+C here to stop the server.
echo.

start "" "%OPEN_URL%"
cd /d "%BACKEND%"
set "PORT=%PORT%"
set "HOST=%HOST%"
%PYTHON% server.py

set "EXIT_CODE=%errorlevel%"
echo.
echo [INFO] Server exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%

:FindPython
set "PYTHON="
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    exit /b 0
)
py --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py"
    exit /b 0
)
echo [ERROR] Python is not installed. Please install Python 3.10+.
echo Download: https://www.python.org/downloads/
pause
exit /b 1

:PickPort
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Port %PORT% is already in use. Switching to 8001.
    set "PORT=8001"
)
exit /b 0
