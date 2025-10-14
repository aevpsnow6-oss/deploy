#!/bin/bash
# ILO Document Evaluator - Docker Launcher (Mac/Linux)
# This script starts the application using Docker

echo "========================================"
echo "ILO Document Evaluator (Docker)"
echo "========================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not running"
    echo "Please install Docker Desktop from docker.com"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose is not installed"
    echo "Please install docker-compose"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found"
    echo "Please create .env file with your OPENAI_API_KEY"
    echo "You can copy .env.example and add your API key"
    echo ""
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "Starting Docker containers..."
echo ""

# Start with docker-compose
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start Docker containers"
    exit 1
fi

echo ""
echo "========================================"
echo "Application started successfully!"
echo "========================================"
echo ""
echo "Open your browser to: http://localhost:8501"
echo ""
echo "To stop the application, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
echo ""

# Wait a moment then open browser
sleep 3

# Open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8501 2>/dev/null
fi
