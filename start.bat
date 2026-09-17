@echo off
REM Start the API and the UI together on Windows.
REM
REM   start.bat
REM
REM Opens each service in its own window so you can read the logs and close
REM them independently. Resolves its own directory, so it works no matter
REM which folder you run it from.

setlocal

REM %~dp0 is this script's directory, with a trailing backslash.
cd /d "%~dp0"

if "%API_HOST%"=="" set API_HOST=127.0.0.1
if "%API_PORT%"=="" set API_PORT=8000
if "%UI_PORT%"=="" set UI_PORT=8501

REM Prefer the venv interpreter if one exists, so activation is optional.
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo error: Python was not found on PATH.
    echo        Install Python 3.10+ from https://python.org and tick
    echo        "Add Python to PATH" during setup.
    pause
    exit /b 1
)

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo error: Python 3.10 or newer is required.
    %PYTHON% --version
    echo.
    echo        Recreate the virtual environment with a newer interpreter:
    echo          rmdir /s /q .venv
    echo          py -3.12 -m venv .venv
    echo          .venv\Scripts\activate
    echo          pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "data\support_tickets.csv" (
    echo error: data\support_tickets.csv is missing.
    pause
    exit /b 1
)

if not exist ".env" (
    echo note: no .env found. /health, /meta and /anomalies will work;
    echo       /query needs a provider. Copy .env.example to .env to set one up.
    echo.
)

echo Starting API -^> http://%API_HOST%:%API_PORT%    ^(docs at /docs^)
start "Support Ticket AI - API" cmd /k "%PYTHON% -m uvicorn app.main:app --host %API_HOST% --port %API_PORT%"

REM Give the API a moment so the UI's first health call succeeds.
timeout /t 5 /nobreak >nul

echo Starting UI  -^> http://localhost:%UI_PORT%
start "Support Ticket AI - UI" cmd /k set "API_URL=http://%API_HOST%:%API_PORT%" && %PYTHON% -m streamlit run ui/app.py --server.port %UI_PORT% --server.headless true --browser.gatherUsageStats false"

echo.
echo Both services are starting in separate windows.
echo Close those windows to stop them.
echo.
echo   API: http://%API_HOST%:%API_PORT%/docs
echo   UI:  http://localhost:%UI_PORT%
echo.

endlocal
