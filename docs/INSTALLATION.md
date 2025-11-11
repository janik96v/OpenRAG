# OpenRAG Installation Guide

This guide follows the project conventions specified in CLAUDE.md.

## Prerequisites

- **Anaconda or Miniconda** installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- **Python 3.13** (will be installed via conda)
- **10+ GB free disk space** (for dependencies and embedding models)
- **Internet connection** (for package and model downloads)

## Quick Install (Automated)

The easiest way to set up OpenRAG:

```bash
cd /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG

# Run automated setup script
./setup_environment.sh
```

This script will:
1. ✅ Create conda environment named "OpenRAG"
2. ✅ Install Python 3.12
3. ✅ Install all required dependencies
4. ✅ Install development tools (optional)
5. ✅ Install OpenRAG package in editable mode
6. ✅ Verify installation
7. ✅ Run quick test (optional)

**Time**: ~10-15 minutes (depending on internet speed)

---

## Manual Installation

If you prefer manual control:

### Step 1: Create Conda Environment

Following CLAUDE.md specification:

```bash
# Create new conda environment "OpenRAG" with Python 3.13
conda create -n OpenRAG python=3.12 -y

# Activate the environment
conda activate OpenRAG
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 3: Install Dependencies

#### Option A: Using requirements.txt (Recommended)

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### Option B: Using pyproject.toml

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or install without dev dependencies
pip install -e .
```

### Step 4: Verify Installation

```bash
python -c "
import chromadb
import sentence_transformers
import tiktoken
import pydantic
print('✅ All dependencies installed successfully!')
"
```

---

## Dependency Details

### Core Dependencies (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| mcp | ≥0.1.0 | Model Context Protocol SDK |
| chromadb | ≥0.4.0 | Vector database |
| sentence-transformers | ≥2.2.0 | Embedding models |
| pydantic | ≥2.0.0 | Data validation |
| pydantic-settings | ≥2.0.0 | Configuration management |
| tiktoken | ≥0.5.0 | Token counting |
| torch | ≥2.0.0 | PyTorch (required by sentence-transformers) |

### Development Dependencies (requirements-dev.txt)

| Package | Purpose |
|---------|---------|
| pytest, pytest-asyncio, pytest-cov | Testing |
| ruff | Linting and formatting |
| mypy | Type checking |
| black, isort | Code formatting |

### Download Sizes

- **Core dependencies**: ~2-3 GB
- **Embedding models** (on first use):
  - all-MiniLM-L6-v2: ~80 MB
  - all-mpnet-base-v2: ~420 MB

---

## Configuration

### Step 1: Create .env File

```bash
cp .env.example .env
```

### Step 2: Edit Configuration (Optional)

Open `.env` and customize:

```bash
# ChromaDB Settings
CHROMA_DB_PATH=./chroma_db

# Embedding Model (choose one)
EMBEDDING_MODEL=all-mpnet-base-v2    # Best quality (default)
# EMBEDDING_MODEL=all-MiniLM-L6-v2   # Faster, smaller

# Chunking Configuration
CHUNK_SIZE=400                        # Tokens per chunk
CHUNK_OVERLAP=60                      # Overlap tokens

# Logging
LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR
```

---

## Verify Installation

### Quick Test

```bash
python quick_test.py
```

Expected output:
```
================================================================================
OPENRAG QUICK TEST
================================================================================

📦 Test 1: Testing imports...
✅ All imports successful

⚙️  Test 2: Testing configuration...
✅ Settings loaded

🔪 Test 3: Testing text chunker...
✅ Chunker working

🧮 Test 4: Testing embedding model...
✅ Embedding model loaded

💾 Test 5: Testing vector store...
✅ Vector store initialized

🔄 Test 6: Testing async tools...
✅ Async tools working

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

### Run Test Suite

```bash
pytest tests/ -v
```

---

## Conda Environment Management

### Activate Environment

```bash
conda activate OpenRAG
```

### Deactivate Environment

```bash
conda deactivate
```

### List Installed Packages

```bash
conda list
```

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Remove Environment

```bash
conda env remove -n OpenRAG
```

### Export Environment

```bash
# Export to environment.yml
conda env export > environment.yml

# Export to requirements.txt
pip freeze > requirements-freeze.txt
```

---

## Troubleshooting

### Issue: `conda: command not found`

**Solution**: Ensure conda is in your PATH:

```bash
# Check conda installation
which conda

# If not found, add to PATH (example for zsh)
echo 'export PATH="/opt/anaconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Issue: `mcp` package not found

**Cause**: MCP requires Python 3.10+

**Solution**:
```bash
# Verify Python version
python --version  # Should be 3.13

# If wrong version, recreate environment
conda env remove -n OpenRAG
conda create -n OpenRAG python=3.13 -y
```

### Issue: Embedding model download fails

**Cause**: Network issues or disk space

**Solution**:
```bash
# Check disk space
df -h

# Check internet connection
ping google.com

# Try downloading manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: ChromaDB permission error

**Solution**:
```bash
# Fix permissions
chmod -R 755 ./chroma_db

# Or use different directory
export CHROMA_DB_PATH=/tmp/chroma_db
```

### Issue: Tests fail with import errors

**Solution**:
```bash
# Ensure OpenRAG is installed in editable mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=/path/to/OpenRAG:$PYTHONPATH
```

---

## Platform-Specific Notes

### macOS (Apple Silicon M1/M2/M3)

PyTorch and sentence-transformers have native Apple Silicon support:

```bash
# No special configuration needed
# Will use Metal Performance Shaders automatically
```

### macOS (Intel)

```bash
# Standard installation works
```

### Linux

```bash
# May need additional packages for ChromaDB
sudo apt-get install build-essential python3-dev
```

### Windows

```bash
# Use Anaconda Prompt
conda activate OpenRAG
```

---

## IDE Integration

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "/path/to/anaconda3/envs/OpenRAG/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black"
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Conda Environment
3. Select existing environment: OpenRAG

---

## Next Steps

After successful installation:

1. ✅ **Configure**: Edit `.env` file
2. ✅ **Test**: Run `pytest tests/ -v`
3. ✅ **Try it**: Run `python quick_test.py`
4. ✅ **Use it**: See [TESTING.md](TESTING.md) for usage examples
5. ✅ **Integrate**: See [README.md](README.md) for Claude Desktop setup

---

## Getting Help

- **Installation issues**: Check [Troubleshooting](#troubleshooting) section
- **Usage questions**: See [TESTING.md](TESTING.md)
- **Development**: See [CLAUDE.md](CLAUDE.md)
- **Bug reports**: Open an issue on GitHub

---

**Last Updated**: 2025-11-09
**Tested On**: macOS Sonoma 14.x, Python 3.12, Conda 24.x
