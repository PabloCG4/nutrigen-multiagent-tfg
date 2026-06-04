#!/bin/bash
# System setup script for Nutrigen multi-agent platform

echo "Checking environment prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not added to the system PATH."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: Node.js/npm is not installed or not added to the system PATH."
    exit 1
fi

echo "Initialization started successfully."

echo "Setting up Python virtual environment..."
python3 -m venv .venv

echo "Activating environment and installing backend dependencies..."
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install uvicorn[standard]

echo "Navigating to frontend directory..."
cd frontend
npm install
cd ..

echo "Installation process completed."
echo "Remember to configure the unified .env file in the root directory before running the system."