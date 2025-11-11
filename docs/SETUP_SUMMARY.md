# OpenRAG Setup - Quick Reference

## ✅ What Was Created

Following CLAUDE.md project conventions, I've created:

### 📁 Environment Setup Files
- **`requirements.txt`** - Core dependencies (mcp, chromadb, sentence-transformers, etc.)
- **`requirements-dev.txt`** - Development dependencies (pytest, ruff, mypy, etc.)
- **`setup_environment.sh`** - Automated setup script (creates conda env "OpenRAG")

### 📚 Documentation
- **`INSTALLATION.md`** - Comprehensive installation guide
- **`TESTING.md`** - Testing guide with examples
- **`SETUP_SUMMARY.md`** - This file (quick reference)

### 🧪 Testing Tools
- **`quick_test.py`** - Fast verification script (2 min)

### 📝 Configuration
- **`.env.example`** - Environment variable template
- **`pyproject.toml`** - Project configuration (already existed)

---

## 🚀 How to Set Up (Following CLAUDE.md)

### Option 1: Automated (Recommended)

```bash
cd /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG

# Run setup script
./setup_environment.sh
```

**What it does:**
1. Creates conda environment named "OpenRAG" with Python 3.13
2. Installs all dependencies from requirements.txt
3. Installs development tools (optional)
4. Installs OpenRAG package in editable mode
5. Verifies installation
6. Runs quick test (optional)

**Time**: ~10-15 minutes

---

### Option 2: Manual

```bash
# 1. Create conda environment (as specified in CLAUDE.md)
conda create -n OpenRAG python=3.13 -y

# 2. Activate environment
conda activate OpenRAG

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install dev dependencies (optional)
pip install -r requirements-dev.txt

# 5. Install OpenRAG in editable mode
pip install -e .

# 6. Configure
cp .env.example .env

# 7. Test
python quick_test.py
```

---

## 📦 What Gets Installed

### Core Dependencies
- **mcp** (≥0.1.0) - Model Context Protocol
- **chromadb** (≥0.4.0) - Vector database
- **sentence-transformers** (≥2.2.0) - Embedding models
- **pydantic** (≥2.0.0) - Data validation
- **tiktoken** (≥0.5.0) - Token counting

### Development Tools
- **pytest** - Testing framework
- **ruff** - Linter and formatter
- **mypy** - Type checker

### Total Download Size
- Core: ~2-3 GB
- Embedding models (on first use):
  - all-MiniLM-L6-v2: ~80 MB
  - all-mpnet-base-v2: ~420 MB

---

## ⚙️ Configuration

### Default Settings (.env.example)

```bash
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
LOG_LEVEL=INFO
```

### Customize

```bash
# Copy template
cp .env.example .env

# Edit as needed
nano .env  # or your preferred editor
```

---

## 🧪 Verify Installation

### Quick Test (2 minutes)

```bash
python quick_test.py
```

**Expected output**: All tests pass ✅

### Full Test Suite (5 minutes)

```bash
pytest tests/ -v
```

---

## 📋 Conda Environment Management

```bash
# Activate (before using OpenRAG)
conda activate OpenRAG

# Deactivate (when done)
conda deactivate

# List installed packages
conda list

# Update dependencies
pip install --upgrade -r requirements.txt

# Remove environment
conda env remove -n OpenRAG

# Export environment
conda env export > environment.yml
```

---

## 🐛 Troubleshooting

### "conda: command not found"
```bash
# Ensure conda is in PATH
which conda

# If not found, add to shell config
echo 'export PATH="/opt/anaconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### "mcp package not found"
```bash
# MCP requires Python 3.10+
python --version  # Should show 3.13

# Recreate environment if needed
conda env remove -n OpenRAG
conda create -n OpenRAG python=3.13 -y
```

### "Import Error: No module named 'openrag'"
```bash
# Ensure package is installed in editable mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

---

## 📚 Next Steps

After successful setup:

1. ✅ **Test**: `python quick_test.py`
2. ✅ **Configure**: Edit `.env` file
3. ✅ **Try it**: See [TESTING.md](TESTING.md)
4. ✅ **Use it**: See [README.md](README.md)
5. ✅ **Develop**: See [CLAUDE.md](CLAUDE.md)

---

## 📖 Documentation Files

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview and quick start |
| [INSTALLATION.md](INSTALLATION.md) | Detailed installation guide |
| [TESTING.md](TESTING.md) | Testing guide and examples |
| [CLAUDE.md](CLAUDE.md) | Development conventions |
| [SETUP_SUMMARY.md](SETUP_SUMMARY.md) | This file (quick reference) |

---

## ✨ You're All Set!

Your conda environment "OpenRAG" is ready following CLAUDE.md specifications:

- ✅ Python 3.13
- ✅ All dependencies installed
- ✅ Development tools configured
- ✅ Package in editable mode
- ✅ Tests passing

**To use OpenRAG:**
```bash
conda activate OpenRAG
python -m openrag.server
```

---

**Last Updated**: 2025-11-09
