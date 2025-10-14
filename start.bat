@echo off
REM ILO Document Evaluator - Windows Launcher
REM This script starts the Streamlit application locally

echo ========================================
echo ILO Document Evaluator
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found
    echo Please create .env file with your OPENAI_API_KEY
    echo You can copy .env.example and add your API key
    echo.
    pause
    exit /b 1
)

REM Start Streamlit
echo Starting application...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

start http://localhost:8501
streamlit run oli_v6_deploy.py

REM Deactivate virtual environment on exit
deactivate
