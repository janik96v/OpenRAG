# Lab Journal - OpenRAG Research

This document serves as the centralized knowledge repository for the OpenRAG project, tracking research findings, technical decisions, and implementation insights.

---

## 2025-11-08 - MCP Server Architecture and Implementation

**Context**: Initial research phase for building an MCP (Model Context Protocol) server to enable RAG (Retrieval Augmented Generation) capabilities over personal documents. This research supports the foundational architecture and implementation strategy for OpenRAG.

**Sources**:
- Official MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP Documentation: https://modelcontextprotocol.io/
- DataCamp MCP Tutorial: https://www.datacamp.com/tutorial/mcp-model-context-protocol
- MCP Architecture Deep Dive: https://www.getknit.dev/blog/mcp-architecture-deep-dive-tools-resources-and-prompts-explained
- Error Handling Guide: https://mcpcat.io/guides/error-handling-custom-mcp-servers/
- Microsoft Best Practices: https://github.com/microsoft/mcp-for-beginners/blob/main/08-BestPractices/README.md

**Key Findings**:

### What is MCP?
- Model Context Protocol (MCP) is an open standard launched in November 2024 by Anthropic
- Enables LLMs to dynamically interact with external tools, databases, and APIs through standardized interfaces
- Separates concerns: providing context vs. LLM interaction
- Addresses fundamental challenge of context integration in AI development

### Core MCP Components

#### 1. Tools
- Specific actions your MCP server exposes to clients (like built-in functions or APIs)
- Used when LLMs need to perform work: fetch data, trigger operations, or calculations
- Tools execute logic and return results to the LLM

**Implementation Pattern**:
```python
@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> WeatherData:
    """Get weather for a city - returns structured data."""
    return WeatherData(
        temperature=22.5,
        humidity=45.0,
        condition="sunny",
        wind_speed=5.2
    )
```

#### 2. Resources
- Expose local or remote data to feed into LLM context
- Provide read-only data access (vs. tools which perform actions)
- Support dynamic path parameters for flexible data retrieval

#### 3. Prompts
- Define structured messages and instructions for interacting with LLMs
- Unlike tools (execute logic) or resources (provide data), prompts return predefined message templates
- Initiate consistent model behavior across interactions

### MCP Server Implementation

**Installation**:
```bash
# Recommended approach using uv
uv init mcp-server
cd mcp-server
uv add "mcp[cli]"

# Alternative with pip
pip install "mcp[cli]"
```

**Server Architecture**:
- Python 3.10+ required
- Async-first design for optimal performance
- Strong typing with Pydantic models
- Supports multiple transports: stdio, SSE, HTTP

**Advanced Tool Result Handling**:
```python
def advanced_tool() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="Response")],
        _meta={"hidden": "client-only data"}
    )
```

**Structured Output Support**:
- Pydantic models
- TypedDicts
- Dataclasses
- Primitive types
- Generic types

### Error Handling Best Practices

#### Error Categories

**Protocol-Level Errors**:
- JSON-RPC 2.0 violations
- Malformed JSON, non-existent methods, invalid parameters
- Must respond with standardized error codes

**Application-Level Errors**:
- Business logic failures
- External API errors
- Resource constraints
- Use `isError` flag in tool responses
- Allows LLM to understand and potentially recover

#### Logging Requirements
- MCP servers MUST write only JSON-RPC messages to stdout
- All logs and debugging output MUST go to stderr
- Critical for proper protocol communication

#### Recovery Patterns
- Provide context-aware responses that help LLM understand failures
- Include specific error details and suggested corrections
- Example: For SQL syntax error, include the issue and suggest fixes

### Testing Strategy
1. Resource Handlers: Test each handler's logic independently
2. Tool Implementations: Verify behavior with various inputs
3. Prompt Templates: Ensure templates render correctly
4. Schema Validation: Test parameter validation logic
5. Error Handling: Verify error responses for invalid inputs

### Security Considerations
- MUST validate all prompt inputs and outputs to prevent injection attacks
- Validate tool inputs to prevent harmful command execution
- Check URIs in resources to prevent unauthorized file access
- Implement guardrails on prompts to prevent unsafe operations
- Never expose sensitive data without proper access controls

### Design Best Practices

**Tool Design**:
- Each tool should have a clear, focused purpose
- Develop specialized tools rather than monolithic ones
- Always validate inputs against expected formats
- Handle errors gracefully with helpful messages

**Prompt Best Practices**:
- Document required arguments and expected input types
- Use clear, actionable names (e.g., `summarize-errors` not `get-summarized-error-log-output`)
- Validate all required arguments upfront
- Provide helpful suggestions for missing/improper inputs
- Build in graceful error handling

**Implementation Notes**:

**For OpenRAG MCP Server**:
1. Use FastMCP for simplified server implementation
2. Implement tools for document ingestion, search, and retrieval
3. Expose resources for collection metadata and document listings
4. Define prompts for common RAG query patterns
5. Ensure all file operations validate paths for security
6. Log all operations to stderr for debugging
7. Return structured results using Pydantic models

**Example GitHub Repositories**:
- Official SDK: https://github.com/modelcontextprotocol/python-sdk
- Simple Tutorial: https://github.com/ruslanmv/Simple-MCP-Server-with-Python
- FastMCP (Pythonic): https://github.com/jlowin/fastmcp
- Python Template: https://github.com/sontallive/mcp-server-python-template
- Official Examples: https://github.com/modelcontextprotocol/servers

**Deployment Considerations**:
- Development mode: Use stdio transport for testing with Claude Desktop
- Production mode: Consider HTTP/SSE transport for web integration
- Package as Python module with proper entry points
- Use environment variables for configuration (API keys, paths)
- Implement graceful shutdown handling
- Monitor stderr logs for operational insights

**Tags**: #MCP #server-architecture #Python #protocol #error-handling #best-practices

---

## 2025-11-08 - ChromaDB for RAG Implementation

**Context**: Researching ChromaDB as the vector database backend for OpenRAG's document storage and retrieval system. ChromaDB provides local-first, persistent storage with embedding support.

**Sources**:
- ChromaDB Official Docs: https://docs.trychroma.com/docs/overview/introduction
- ChromaDB Cookbook - Persistent Client: https://docs.trychroma.com/docs/run-chroma/persistent-client
- ChromaDB Cookbook - Storage Layout: https://cookbook.chromadb.dev/core/storage-layout/
- ChromaDB Cookbook - Running Chroma: https://cookbook.chromadb.dev/running/running-chroma/
- RAG Advanced Techniques: https://medium.com/@sulaiman.shamasna/rag-i-advanced-techniques-with-chroma-dd8c7c08d000
- Chroma Chunking Research: https://research.trychroma.com/evaluating-chunking

**Key Findings**:

### ChromaDB Overview
- Open-source AI application database (Apache 2.0 license)
- Designed to make knowledge and skills "pluggable for LLMs"
- Comprehensive solution for embedding and retrieval tasks
- Runs as embedded database or client-server

### Core Capabilities
1. **Retrieval Features**:
   - Store embeddings and metadata
   - Vector search (semantic similarity)
   - Full-text search
   - Document storage
   - Metadata filtering
   - Multi-modal retrieval

2. **Architecture**:
   - Multiple operational modes: ephemeral, persistent, client-server, cloud
   - Python and JavaScript/TypeScript client SDKs
   - Extensive language support (Ruby, Java, Go, C#, Rust, etc.)

### Installation and Setup

**Installation**:
```bash
pip install chromadb
```

**Persistent Client Setup**:
```python
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

# Basic persistent client
client = chromadb.PersistentClient(path="/path/to/data")

# With additional configuration
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
```

### Storage Configuration

**Path Configuration**:
- Path must be local path on machine where Chroma runs
- If path doesn't exist, it will be created
- Path can be relative or absolute
- Default: `./chroma` in current working directory

**Storage Structure**:
- Creates directory structure under persistent directory
- Contains `chroma.sqlite3` file for metadata
- Stores vector data and embeddings locally

### Collection Management

**Creating Collections**:
```python
# Get or create collection
collection = client.get_or_create_collection(name="documents")

# Best practice: specify embedding function at creation
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function  # Avoid default
)
```

**Default Behavior**:
- If no embedding function specified, uses sentence-transformer by default
- BEST PRACTICE: Always specify embedding function explicitly for consistency

### RAG Implementation Best Practices

**Embedding Storage**:
- Text chunks embedded and stored via embedding model
- ChromaDB handles vector storage and retrieval efficiently
- Supports metadata for filtering and enhanced retrieval

**Query Patterns**:
```python
# Query with metadata filtering
results = collection.query(
    query_texts=["search query"],
    n_results=10,
    where={"category": "documentation"}  # Metadata filter
)

# Multi-modal queries
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],  # Pre-computed embeddings
    n_results=5
)
```

### Use Cases for OpenRAG
- Local-first document storage (privacy-preserving)
- Semantic search over personal documents
- Knowledge retrieval without external dependencies
- Development and testing with embedded mode
- Production deployment with client-server mode

**Implementation Notes**:

**For OpenRAG**:
1. Use PersistentClient for local storage
2. Create separate collections for different document types
3. Specify embedding function explicitly (sentence-transformers)
4. Store rich metadata (filename, date, document type, tags)
5. Implement metadata filtering for scoped searches
6. Path configuration via environment variables
7. Backup strategy for chroma_db directory

**Storage Considerations**:
- SQLite backend limits concurrent writes (single writer)
- For high-concurrency, consider client-server mode
- Embeddings stored efficiently in columnar format
- Plan for backup of persistent directory
- Monitor disk usage as collection grows

**Performance Tips**:
- Batch insertions for better performance
- Use metadata filtering to reduce search space
- Consider collection per document type for isolation
- Monitor query latency and optimize chunk size accordingly

**Tags**: #ChromaDB #vector-database #RAG #local-storage #persistent-client #embeddings

---

## 2025-11-08 - Embedding Models Comparison and Selection

**Context**: Evaluating embedding models for OpenRAG, balancing speed vs. quality for local document retrieval. Key consideration: models must run efficiently on consumer hardware while providing good semantic search quality.

**Sources**:
- Sentence Transformers Docs: https://www.sbert.net/
- Model Comparison: https://milvus.io/ai-quick-reference/what-are-some-popular-pretrained-sentence-transformer-models-and-how-do-they-differ-for-example-allminilml6v2-vs-allmpnetbasev2
- Instructor-XL Hub: https://huggingface.co/hkunlp/instructor-xl
- Instructor Embedding: https://instructor-embedding.github.io/
- Performance Analysis: https://www.pragnakalp.com/open-source-embedding-models-which-one-performs-best/
- Pretrained Models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

**Key Findings**:

### Sentence Transformers Overview

**Model Types**:
1. Sentence Transformer (embedding models) - our focus
2. Cross Encoder (reranker models)
3. Sparse Encoder (sparse embedding models)

**Ecosystem**:
- Over 10,000 pre-trained models on Hugging Face
- Multilingual support
- Task-specific fine-tuning capabilities
- Efficiency optimizations: ONNX, OpenVINO, quantization, multi-GPU

### Model Comparison Matrix

#### all-MiniLM-L6-v2 (Fast, Lightweight)

**Architecture**:
- Distilled model with 6 layers (vs. 12 in full models)
- Hidden dimensions: 384
- Model size: ~80 MB
- Optimized for speed

**Performance**:
- Processing speed: 3-6 seconds for typical document batches
- 5x faster than all-mpnet-base-v2
- MTEB benchmark score: ~56 (average across tasks)
- Embedding dimension: 384

**Best For**:
- Latency-critical applications
- Hardware-limited environments
- High-throughput document processing
- Real-time search scenarios
- Development and prototyping

**Limitations**:
- Lower accuracy compared to larger models
- May miss subtle semantic nuances
- Not ideal for domain-specific terminology

#### all-mpnet-base-v2 (Balanced, High Quality)

**Architecture**:
- Based on MPNet architecture
- Combines BERT's masked language modeling + XLNet's permutation pretraining
- 12 layers
- Hidden dimensions: 768
- Model size: ~420 MB

**Performance**:
- Processing speed: 30-50 seconds for typical document batches
- Best quality among base models
- Higher MTEB benchmark scores than MiniLM
- Embedding dimension: 768
- Trained on 1+ billion training pairs

**Best For**:
- Accuracy-critical applications
- Legal document analysis
- Academic research
- Medical/scientific text
- Production deployments where quality > speed

**Limitations**:
- 5x slower than MiniLM
- Higher memory requirements
- More computational resources needed

#### instructor-xl (Instruction-Tuned, Versatile)

**Architecture**:
- Based on instructor-embedding framework
- Instruction-finetuned for task-specific embeddings
- XL variant (largest in series)
- Native support in Sentence Transformers

**Performance**:
- State-of-the-art on 70+ diverse embedding tasks
- Better than instructor-large
- Task-specific embeddings without additional fine-tuning
- VRAM usage at batch 10: ~6,846 MB
- Optimal batch size: 5-10 for compute efficiency

**Unique Features**:
- Takes task instructions as input
- Generates embeddings based on task context
- Flexible for diverse applications without retraining
- Instructions not included in pooling step

**Example Usage**:
```python
# Task-specific embedding with instruction
instruction = "Represent the document for retrieval:"
embedding = model.encode([[instruction, document_text]])
```

**Best For**:
- Multi-domain applications
- Task-aware semantic search
- Applications requiring flexible embedding behavior
- Research projects with diverse use cases

**Limitations**:
- Highest resource requirements
- Requires instruction design
- May be overkill for simple retrieval tasks

### Performance Summary Table

| Model | Speed | Quality | Size | VRAM | Best Use Case |
|-------|-------|---------|------|------|---------------|
| all-MiniLM-L6-v2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 80 MB | Low | Fast prototyping, real-time |
| all-mpnet-base-v2 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 420 MB | Medium | Production quality search |
| instructor-xl | ⭐⭐ | ⭐⭐⭐⭐⭐ | Large | High (~7 GB) | Multi-task, research |

### Recommendations for OpenRAG

**Development Phase**:
- Start with **all-MiniLM-L6-v2**
- Fastest iteration cycles
- Sufficient quality for testing chunking strategies
- Minimal hardware requirements

**Production Deployment**:
- Default: **all-mpnet-base-v2**
- Best balance of quality and performance
- Proven reliability across diverse document types
- Acceptable latency for most use cases

**Advanced/Research Use**:
- Consider **instructor-xl** for:
  - Multi-domain document collections
  - Task-specific retrieval scenarios
  - When users can tolerate higher latency
  - Sufficient hardware available (GPU with 8+ GB VRAM)

**Implementation Strategy**:
```python
from sentence_transformers import SentenceTransformer

# Configuration via environment variable
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Load model
embedding_model = SentenceTransformer(MODEL_NAME)

# For instructor models
if "instructor" in MODEL_NAME:
    instruction = "Represent the document for retrieval:"
    embeddings = embedding_model.encode([[instruction, text]])
else:
    embeddings = embedding_model.encode([text])
```

**Implementation Notes**:

**Model Selection Criteria**:
1. User hardware capabilities (CPU vs. GPU, RAM available)
2. Document corpus size (affects total processing time)
3. Query latency requirements (real-time vs. batch)
4. Domain specificity (general vs. specialized)
5. Update frequency (how often to re-embed documents)

**Optimization Techniques**:
- Use ONNX backend for faster inference
- Batch processing for bulk embeddings
- Cache embeddings to avoid recomputation
- Consider model quantization for production

**Migration Path**:
- Design abstraction layer for embedding models
- Allow runtime model switching
- Store model name with embeddings for tracking
- Implement reindexing workflow for model upgrades

**Tags**: #embeddings #sentence-transformers #model-selection #performance #all-MiniLM #all-mpnet #instructor-xl

---

## 2025-11-08 - Text Chunking Strategies for RAG

**Context**: Researching optimal text chunking strategies for document ingestion in OpenRAG. Chunking quality directly impacts retrieval accuracy and LLM response quality.

**Sources**:
- Stack Overflow RAG Chunking: https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/
- Firecrawl 2025 Guide: https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025
- Weaviate Best Practices: https://weaviate.io/blog/chunking-strategies-for-rag
- Multimodal.dev Semantic: https://www.multimodal.dev/post/semantic-chunking-for-rag
- Databricks Guide: https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
- 11 Strategies: https://masteringllm.medium.com/11-chunking-strategies-for-rag-simplified-visualized-df0dbec8e373
- ChromaDB Research: https://research.trychroma.com/evaluating-chunking

**Key Findings**:

### Chunking Strategy Overview

**Why Chunking Matters**:
- Embedding models have maximum token limits (e.g., 8,191 tokens for text-embedding-ada-002)
- Smaller chunks provide more focused context to LLMs
- Chunking quality directly affects retrieval precision
- Wrong strategy can create up to 9% gap in recall performance

### Core Chunking Strategies

#### 1. Fixed-Size Chunking

**Description**:
- Split text into fixed-length chunks (e.g., 500 tokens)
- Simplest approach to implement
- Ignores document structure and semantic boundaries

**Implementation**:
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  # 10% overlap
    separator="\n"
)
chunks = splitter.split_text(document)
```

**Pros**:
- Easy to implement
- Predictable chunk sizes
- Fast processing
- Consistent token usage

**Cons**:
- Breaks semantic units (sentences, paragraphs)
- May split critical information
- Ignores document structure
- Not context-aware

**Best For**:
- Quick prototyping
- Homogeneous documents
- When speed > quality

#### 2. RecursiveCharacterTextSplitter (INDUSTRY STANDARD)

**Description**:
- Attempts to split on semantic boundaries hierarchically
- Default separators: `["\n\n", "\n", ".", "?", "!", " ", ""]`
- Recursively tries separators until chunk size achieved

**Implementation**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Recommended: 400-512 tokens
    chunk_overlap=50,  # 10-20% overlap
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)
chunks = splitter.split_text(document)
```

**2024 Industry Consensus**:
- **Start here**: 400-512 tokens with 10-20% overlap
- Most widely adopted baseline strategy
- Good balance of simplicity and quality

**Pros**:
- Respects semantic boundaries where possible
- Configurable separators for different document types
- Better than fixed-size for most use cases
- Well-tested and reliable

**Cons**:
- Still may break semantic units
- Doesn't understand content meaning
- Fixed separator hierarchy may not suit all documents

**Best For**:
- General-purpose RAG applications
- Mixed document types
- Baseline implementation before optimization

#### 3. Semantic Chunking

**Description**:
- Analyzes semantic similarity between consecutive sentences
- Creates chunks where topic shifts occur
- Groups sentences based on embedding similarity

**Implementation**:
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"  # or "standard_deviation"
)
chunks = splitter.split_text(document)
```

**Pros**:
- Preserves semantic coherence
- Context-aware chunking
- Better for complex documents
- Reduces fragmentation

**Cons**:
- Requires embedding model (cost/latency)
- More computational overhead
- Variable chunk sizes
- May produce very large or small chunks

**Best For**:
- High-quality retrieval requirements
- Complex documents with topic shifts
- When budget allows for processing cost

#### 4. ClusterSemanticChunker (OPTIMAL QUALITY)

**Description**:
- Produces globally optimal chunks
- Maximizes sum of cosine similarities within chunks
- Subject to user-specified maximum length

**Performance**:
- ChromaDB research recommends for maximum semantic coherence
- 200-400 token chunks optimal
- Best retrieval quality in benchmarks

**Pros**:
- Highest semantic coherence
- Optimal information preservation
- Best retrieval accuracy

**Cons**:
- Most computationally expensive
- Requires careful tuning
- May be overkill for simple documents

**Best For**:
- Premium applications requiring best quality
- Academic/research documents
- Legal or medical texts

#### 5. Page-Level Chunking

**Description**:
- Treats each page as a chunk
- Preserves document structure

**Performance**:
- NVIDIA 2024 benchmark: highest accuracy (0.648)
- Lowest standard deviation (0.107)
- Best for document-native content (PDFs)

**Pros**:
- Natural document boundaries
- Preserves page context
- Simple to implement for PDFs

**Cons**:
- Variable chunk sizes
- Pages may exceed token limits
- Only works for paginated content

**Best For**:
- PDF documents
- When page context is important
- Academic papers, reports

### Overlap Recommendations (2024 Industry Standard)

**Consensus**:
- **10-20% overlap** as starting point
- For 500-token chunks: 50-100 token overlap
- Ensures complete thoughts captured across boundaries

**Why Overlap Matters**:
- Reduces fragmentation problems
- If key sentence split across chunks, overlap ensures both contain complete thought
- Prevents information loss at boundaries

**Typical Configurations**:
```python
# Conservative (general purpose)
chunk_size=500
chunk_overlap=50  # 10%

# Moderate (recommended baseline)
chunk_size=400
chunk_overlap=60  # 15%

# Aggressive (maximum context preservation)
chunk_size=512
chunk_overlap=100  # 20%
```

### 2024 Benchmark Results

**NVIDIA Benchmark (5 datasets, 7 strategies)**:
1. Page-level: 0.648 accuracy, 0.107 std dev
2. Semantic chunking: High quality, variable performance
3. RecursiveCharacter: 0.580-0.620 range
4. Fixed-size: 0.550-0.600 range

**Key Insight**: Wrong strategy creates up to 9% gap in recall performance

### Practical Recommendations for OpenRAG

**Phase 1: Baseline (Start Here)**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,  # 15% overlap
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)
```
- Fastest to implement
- Good baseline performance
- Easy to measure improvement against

**Phase 2: Optimization (If Needed)**:
```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile"
)
```
- Move here if baseline metrics show need
- Budget must allow for embedding costs
- Monitor improvement in retrieval quality

**Phase 3: Premium (For Specific Use Cases)**:
- ClusterSemanticChunker for maximum quality
- Page-level for PDF-heavy collections
- Document-type-specific strategies

### Implementation Strategy

**Hybrid Approach**:
```python
def chunk_document(doc, doc_type):
    if doc_type == "pdf":
        # Page-level chunking
        return chunk_by_page(doc)
    elif doc_type == "code":
        # Language-aware chunking
        return chunk_by_ast(doc)
    else:
        # Recursive character splitting
        return recursive_chunk(doc, size=400, overlap=60)
```

**Metrics to Track**:
1. Average chunk size (tokens)
2. Chunk size distribution (variance)
3. Retrieval precision@k
4. Retrieval recall@k
5. User satisfaction with results

**Tuning Process**:
1. Start with RecursiveCharacter 400/60
2. Measure baseline retrieval quality
3. Experiment with chunk size (200-600 range)
4. Adjust overlap (10-25% range)
5. Test semantic chunking if budget allows
6. A/B test strategies on real queries

**Implementation Notes**:

**For OpenRAG**:
1. Default: RecursiveCharacterTextSplitter (400 tokens, 15% overlap)
2. Make strategy configurable per collection
3. Store chunking metadata with each chunk (strategy, params)
4. Implement reindexing workflow for strategy changes
5. Log chunk size statistics for monitoring
6. Consider document-type-specific strategies

**Gotchas**:
- Token counting vs. character counting (use tiktoken for accuracy)
- Separator choice matters for document type
- Very small chunks (< 100 tokens) often hurt performance
- Very large chunks (> 1000 tokens) may exceed context windows
- Overlap creates storage overhead (budget accordingly)

**Future Research**:
- Agentic chunking (LLM-assisted)
- Multi-representation chunking
- Graph-based chunking
- Adaptive chunking based on query patterns

**Tags**: #chunking #RAG #text-splitting #RecursiveCharacter #semantic-chunking #overlap #best-practices

---

## Summary of Research Findings

### Critical Decisions for OpenRAG

**MCP Server**:
- Use official Python SDK with FastMCP
- Implement tools for document operations (ingest, search, delete)
- Expose resources for collection metadata
- Define prompts for common query patterns
- Strict error handling with stderr logging
- Security: validate all inputs, check file paths

**Vector Database**:
- ChromaDB PersistentClient for local storage
- Path via environment variable
- Explicit embedding function specification
- Metadata-rich storage for filtering
- Separate collections by document type

**Embedding Model**:
- Development: all-MiniLM-L6-v2 (fast iteration)
- Production: all-mpnet-base-v2 (quality balance)
- Advanced: instructor-xl (multi-domain)
- Configurable via environment variable
- Abstract model interface for swapping

**Chunking Strategy**:
- Baseline: RecursiveCharacterTextSplitter
- Chunk size: 400 tokens
- Overlap: 15% (60 tokens)
- Make strategy configurable
- Store chunking metadata
- Plan for reindexing

### Next Steps

1. **Create PRP (Project Research Plan)**:
   - Define OpenRAG architecture
   - Specify MCP tool interfaces
   - Design ChromaDB schema
   - Plan configuration system

2. **Prototype Implementation**:
   - Minimal MCP server with one tool
   - ChromaDB integration test
   - Embedding model comparison
   - Chunking strategy validation

3. **Evaluation Framework**:
   - Define retrieval quality metrics
   - Create test document corpus
   - Establish baseline performance
   - Plan A/B testing infrastructure

**Tags**: #summary #decisions #architecture #next-steps

---

## Research Backlog

**Future Research Topics**:
1. Advanced RAG techniques (contextual RAG, graph RAG)
2. Reranking models for improved retrieval
3. Query expansion strategies
4. Hybrid search (vector + full-text)
5. Multi-modal embeddings (text + images)
6. Incremental indexing strategies
7. Deployment options (Docker, cloud)
8. Cost optimization for large corpora
9. Privacy-preserving RAG techniques
10. Evaluation datasets and benchmarks

**Open Questions**:
1. Optimal metadata schema for diverse document types?
2. How to handle very large documents (books, technical manuals)?
3. Should we support multiple embedding models simultaneously?
4. Query preprocessing and optimization strategies?
5. How to handle updates to existing documents?
6. Rate limiting and resource management for MCP server?

**Tags**: #backlog #future-research #open-questions

---

*Last Updated: 2025-11-08*
*Researcher: Claude (Research Agent)*
*Status: Initial comprehensive research complete*
