#!/bin/bash
# ILO Document Evaluator - Mac/Linux Launcher
# This script starts the Streamlit application locally

echo "========================================"
echo "ILO Document Evaluator"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! pip show streamlit &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found"
    echo "Please create .env file with your OPENAI_API_KEY"
    echo "You can copy .env.example and add your API key"
    echo ""
    exit 1
fi

# Start Streamlit
echo "Starting application..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Open browser (works on Mac, adjust for Linux if needed)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8501 2>/dev/null
fi

# Start Streamlit
streamlit run oli_v6_deploy.py

# Deactivate virtual environment on exit
deactivate
