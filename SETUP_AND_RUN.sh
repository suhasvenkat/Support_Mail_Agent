#!/bin/bash

# SupportMailAgent - Complete Setup and Run Script
# This script will:
# 1. Create a Python virtual environment
# 2. Install all dependencies
# 3. Load the FAISS knowledge base
# 4. Start the server

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════╗"
echo "║    SupportMailAgent - Setup & Run                      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Create virtual environment
echo "📦 Step 1: Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Step 2: Activate virtual environment
echo ""
echo "🔌 Step 2: Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Step 3: Install dependencies
echo ""
echo "📚 Step 3: Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Step 4: Verify .env file
echo ""
echo "🔑 Step 4: Checking configuration..."
if [ -f ".env" ]; then
    echo "✓ .env file found"
else
    echo "✗ ERROR: .env file not found"
    echo "Please create .env with your OpenAI API key"
    exit 1
fi

# Step 5: Load knowledge base
echo ""
echo "📚 Step 5: Loading FAISS knowledge base..."
python3 cli_kb_manager.py load
echo ""

# Step 6: Ready message
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║          ✅ SETUP COMPLETE!                            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Starting the server..."
echo ""
echo "The application will be available at:"
echo "   🌐 http://localhost:8000"
echo ""
echo "API Documentation:"
echo "   📖 http://localhost:8000/docs"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""

# Step 7: Start server
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
