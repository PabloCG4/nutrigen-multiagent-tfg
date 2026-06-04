@echo off
rem System setup script for Nutrigen multi-agent platform

echo Checking environment prerequisites...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not added to the system PATH.
    exit /b 1
)

where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Node.js/npm is not installed or not added to the system PATH.
    exit /b 1
)

echo Initialization started successfully.

echo Setting up Python virtual environment...
python -m venv .venv

echo Activating environment and installing backend dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install uvicorn[standard]

echo Navigating to frontend directory...
cd frontend
call npm install
cd ..

echo Installation process completed.
echo Remember to configure the unified .env file in the root directory before running the system.
pause