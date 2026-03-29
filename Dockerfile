FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install CPU-only PyTorch first to avoid pulling large CUDA packages (~1.5GB saved)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Install the package itself (no deps, already installed above)
RUN pip install --no-cache-dir -e . --no-deps

# Cache directory for HuggingFace embedding models (persisted via volume)
ENV HF_HOME=/app/.cache/huggingface

# ChromaDB data directory (persisted via volume)
ENV CHROMA_DB_PATH=/app/chroma_db

CMD ["python", "-m", "openrag.server"]
