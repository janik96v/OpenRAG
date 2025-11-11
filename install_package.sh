#!/bin/bash
# Script to install OpenRAG package in development mode

set -e  # Exit on error

echo "🔍 Checking for OpenRAG conda environment..."

# Check if OpenRAG environment exists
if ! conda env list | grep -q "^OpenRAG "; then
    echo "❌ OpenRAG conda environment not found!"
    echo "📦 Creating OpenRAG environment with Python 3.12..."
    conda create -n OpenRAG python=3.12 -y
fi

echo "✅ OpenRAG environment found"
echo "📦 Installing package in editable mode..."

# Activate environment and install
eval "$(conda shell.bash hook)"
conda activate OpenRAG

# Install the package in editable mode
pip install -e ".[dev]"

echo "✅ Installation complete!"
echo ""
echo "To use the package, activate the environment:"
echo "  conda activate OpenRAG"
