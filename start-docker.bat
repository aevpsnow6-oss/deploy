@echo off
REM ILO Document Evaluator - Docker Launcher (Windows)
REM This script starts the application using Docker

echo ========================================
echo ILO Document Evaluator (Docker)
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop from docker.com
    echo.
    pause
    exit /b 1
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

REM Create data directory if it doesn't exist
if not exist "data\" mkdir data

echo Starting Docker containers...
echo.

REM Start with docker-compose
docker-compose up -d

if errorlevel 1 (
    echo ERROR: Failed to start Docker containers
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Application started successfully!
echo ========================================
echo.
echo Open your browser to: http://localhost:8501
echo.
echo To stop the application, run: docker-compose down
echo To view logs, run: docker-compose logs -f
echo.

timeout /t 3 /nobreak >nul
start http://localhost:8501

pause
