#!/bin/bash

# Quick Start Script for DriveSpatial AV Data API

echo "========================================"
echo "DriveSpatial AV Data API - Quick Start"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check database

# Start the server
echo ""
echo "========================================"
echo "Starting FastAPI server..."
echo "========================================"
echo ""
echo "API will be available at:"
echo "  - Main: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
