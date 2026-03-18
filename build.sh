#!/bin/bash
set -e

echo "🔨 Building Support Mail Agent API..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete!"
