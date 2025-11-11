#!/bin/bash
# Setup script for OpenRAG conda environment
# Following CLAUDE.md specifications

set -e  # Exit on error

echo "=================================="
echo "OpenRAG Environment Setup"
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check Python version requirement
PYTHON_VERSION="3.13"
echo "✓ Conda found"

# Check if environment already exists
if conda env list | grep -q "^OpenRAG "; then
    echo ""
    echo "⚠️  Warning: OpenRAG environment already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n OpenRAG -y
    else
        echo "Aborting. Use 'conda activate OpenRAG' to use existing environment."
        exit 0
    fi
fi

# Create new conda environment
echo ""
echo "📦 Creating conda environment 'OpenRAG' with Python $PYTHON_VERSION..."
conda create -n OpenRAG python=$PYTHON_VERSION -y

# Activate environment
echo ""
echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OpenRAG

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo ""
echo "📚 Installing core dependencies..."
echo "   (This may take several minutes, especially for PyTorch and sentence-transformers)"

# Install requirements
pip install -r requirements.txt

# Install development dependencies
echo ""
read -p "Install development dependencies? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "📚 Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install package in editable mode
echo ""
echo "📦 Installing OpenRAG in editable mode..."
pip install -e .

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import chromadb
    print('✓ chromadb installed')
except ImportError as e:
    print(f'✗ chromadb not found: {e}')

try:
    import sentence_transformers
    print('✓ sentence-transformers installed')
except ImportError as e:
    print(f'✗ sentence-transformers not found: {e}')

try:
    import tiktoken
    print('✓ tiktoken installed')
except ImportError as e:
    print(f'✗ tiktoken not found: {e}')

try:
    import pydantic
    print('✓ pydantic installed')
except ImportError as e:
    print(f'✗ pydantic not found: {e}')

try:
    import openrag
    print('✓ openrag package installed')
except ImportError as e:
    print(f'✗ openrag not found: {e}')
"

# Run quick test if requested
echo ""
read -p "Run quick test to verify setup? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "🧪 Running quick test..."
    python quick_test.py
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate OpenRAG"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and configure"
echo "  2. Run tests: pytest tests/ -v"
echo "  3. Start server: python -m openrag.server"
echo ""
echo "See README.md and TESTING.md for more information."
