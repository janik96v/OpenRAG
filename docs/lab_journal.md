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

## 2025-11-12 - Contextual RAG Implementation Patterns

**Context**: Research for implementing Anthropic's Contextual Retrieval technique in OpenRAG MCP server. Contextual RAG addresses the fundamental problem of traditional RAG systems: chunk boundaries destroy context, leading to retrieval failures. This research informs the implementation strategy for PRP-002 (Contextual RAG support).

**Sources**:
- Anthropic Official Announcement: https://www.anthropic.com/news/contextual-retrieval
- Together.ai Implementation Guide: https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic
- Analytics Vidhya Analysis: https://www.analyticsvidhya.com/blog/2024/11/anthropics-contextual-rag/
- Instructor Async Implementation: https://python.useinstructor.com/blog/2024/09/26/implementing-anthropics-contextual-retrieval-with-async-processing/
- DataCamp Tutorial: https://www.datacamp.com/tutorial/contextual-retrieval-anthropic
- Medium Deep Dive: https://medium.com/@odhitom09/the-most-effective-rag-approach-to-date-anthropics-contextual-retrieval-and-hybrid-search-8dc2af5cb970
- Pluralsight Guide: https://www.pluralsight.com/resources/blog/ai-and-data/how-to-implement-contextual-retrieval

**Key Findings**:

### What is Contextual Retrieval?

**Core Problem**:
- Traditional RAG splits documents into chunks without preserving context
- Individual chunks lack sufficient context when retrieved in isolation
- Results in retrieval failure rates of ~5.7% in baseline systems

**Solution**:
- Chunk augmentation technique that uses an LLM to prepend context to each chunk
- Context situates the chunk within the overall document
- Reduces retrieval failures by 35-67% depending on techniques combined

### Performance Results

**Contextual Embeddings alone**:
- Reduced top-20-chunk retrieval failure rate by 35% (5.7% → 3.7%)

**Contextual Embeddings + Contextual BM25**:
- Reduced failure rate by 49% (5.7% → 2.9%)

**Full Stack (Contextual Embeddings + BM25 + Reranking)**:
- Reduced failure rate by 67% (5.7% → 1.9%)

### Implementation Steps

#### 1. Chunk Creation
```python
# Recommended chunk size for contextual retrieval
chunk_size = 800  # tokens (vs. 400 for traditional RAG)
context_size = 50-100  # tokens of prepended context
```

#### 2. Context Generation Prompt

Anthropic's exact prompt template:
```python
CONTEXTUAL_PROMPT = """<document>
{WHOLE_DOCUMENT}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""
```

**Key Characteristics**:
- Uses XML-style tags for clear structure
- Includes entire document for full context
- Requests short, succinct context (50-100 tokens)
- Emphasizes retrieval optimization purpose
- Instructs to return ONLY context, nothing else

#### 3. Async Processing for Scale

**Concurrent Chunk Processing Pattern**:
```python
import asyncio

async def generate_contextual_chunks(document: str, chunks: List[str]) -> List[Dict]:
    """Process all chunks concurrently for efficiency."""
    tasks = [
        generate_context(document, chunk)
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks)
    return results

async def generate_context(document: str, chunk: str) -> Dict[str, str]:
    """Generate context for a single chunk."""
    prompt = CONTEXTUAL_PROMPT.format(
        WHOLE_DOCUMENT=document,
        CHUNK_CONTENT=chunk
    )
    context = await llm_client.generate(prompt)

    # Prepend context to chunk
    contextualized_chunk = f"{context}\n\n{chunk}"

    return {
        "original_chunk": chunk,
        "context": context,
        "contextualized_chunk": contextualized_chunk
    }
```

#### 4. Hybrid Search Architecture

**Two Parallel Indices**:
```python
# Vector index (semantic search)
vector_collection = chroma_client.get_or_create_collection(
    name="contextual_embeddings",
    embedding_function=embedding_model
)

# BM25 keyword index (lexical search)
from rank_bm25 import BM25Okapi

bm25_index = BM25Okapi([
    chunk.split() for chunk in contextualized_chunks
])
```

#### 5. Reciprocal Rank Fusion (RRF)

**Combining Search Results**:
```python
def reciprocal_rank_fusion(*ranking_lists, k=60):
    """
    Combine multiple ranking lists using RRF algorithm.

    Args:
        *ranking_lists: Variable number of ranking lists
        k: Constant for RRF formula (default: 60)

    Returns:
        Fused ranking with combined scores
    """
    rrf_scores = defaultdict(float)

    for ranking_list in ranking_lists:
        for rank, item_id in enumerate(ranking_list, start=1):
            # RRF formula: score = 1 / (k + rank)
            rrf_scores[item_id] += 1 / (k + rank)

    # Sort by score descending
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# Usage
vector_results = vector_search(query, top_k=150)
bm25_results = bm25_search(query, top_k=150)
fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
```

#### 6. Optional Reranking

**Final Refinement Step**:
```python
# Retrieve top 150 from hybrid search
top_150 = fused_results[:150]

# Rerank to get best 20
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranked = reranker.rank(query, [chunk for _, chunk in top_150])
top_20 = reranked[:20]
```

### Implementation Considerations

#### Cost Optimization with Prompt Caching

**The Problem**:
- Each chunk requires full document as context
- For 1000 chunks, full document sent 1000 times
- Extremely expensive without optimization

**The Solution**:
```python
# Anthropic's prompt caching reduces costs by ~90%
# Cache the document portion of the prompt
response = anthropic.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"<document>{whole_document}</document>",
                "cache_control": {"type": "ephemeral"}  # Cache this part
            },
            {
                "type": "text",
                "text": f"<chunk>{chunk}</chunk>\n\nPlease give context..."
            }
        ]
    }]
)
```

**Caching Benefits**:
- First chunk: Full cost
- Subsequent chunks: ~90% cost reduction
- Cache valid for 5 minutes
- Dramatically reduces cost at scale

#### Recommended Chunk Size

**Key Change from Traditional RAG**:
- Traditional RAG: 400-512 tokens per chunk
- Contextual RAG: **~800 tokens per chunk**

**Reasoning**:
- Larger chunks offset context overhead
- Context adds 50-100 tokens
- Final embedded chunk: ~850-900 tokens
- Still within embedding model limits (most support 512-8192 tokens)

#### Number of Retrieved Chunks

**Anthropic's Finding**: 20 chunks is optimal
- More chunks increase relevant information likelihood
- But too many chunks create noise and distraction
- 20 chunks found most performant in experiments

#### Embedding Model Selection

**Best Performers**:
- Gemini embeddings
- Voyage embeddings
- All embedding models benefit, but some more than others

**For OpenRAG**:
- Continue using sentence-transformers for local deployment
- Test with all-mpnet-base-v2 (current default)
- Consider upgrading to Voyage for production if budget allows

### Common Pitfalls and Gotchas

#### 1. Chunk Size Selection
- **Mistake**: Using same chunk size as traditional RAG (400 tokens)
- **Fix**: Increase to ~800 tokens to account for context overhead
- **Reason**: Context adds 50-100 tokens; larger base chunks maintain semantic completeness

#### 2. Cost Explosion Without Caching
- **Mistake**: Not implementing prompt caching
- **Impact**: 10x-100x cost increase for large documents
- **Fix**: Always use prompt caching for the document portion

#### 3. Over-Contextualization
- **Mistake**: Generating verbose context (>100 tokens)
- **Impact**: Dilutes actual chunk content, increases costs
- **Fix**: Explicitly request "short succinct context" in prompt

#### 4. Ignoring Domain Specificity
- **Mistake**: Using generic context prompt for specialized domains
- **Fix**: Customize prompt with domain glossary or terminology
- **Example**: Medical documents might include "Context should use standard medical terminology"

#### 5. Scalability Challenges
- **Mistake**: Processing chunks sequentially
- **Fix**: Use async/concurrent processing (asyncio.gather)
- **Benefit**: 10-100x speedup depending on concurrency

#### 6. Small Knowledge Base Overhead
- **Mistake**: Using contextual retrieval for tiny knowledge bases
- **Fix**: For <200,000 tokens (~500 pages), include entire KB in prompt
- **Reason**: Retrieval overhead not worth it for small corpora

#### 7. Neglecting Hybrid Search
- **Mistake**: Using only semantic search with contextual embeddings
- **Fix**: Implement both vector and BM25 search with RRF
- **Benefit**: Additional 14% failure rate reduction (35% → 49%)

### Implementation Strategy for OpenRAG

#### Phase 1: Core Contextual Embedding (Week 1-2)
```python
class ContextualChunker:
    def __init__(self, llm_client, chunk_size=800, overlap=100):
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def process_document(self, document: str) -> List[ContextualChunk]:
        # 1. Create base chunks
        chunks = self.create_chunks(document)

        # 2. Generate contexts concurrently
        tasks = [
            self.generate_context(document, chunk)
            for chunk in chunks
        ]
        contextualized = await asyncio.gather(*tasks)

        return contextualized

    async def generate_context(self, doc: str, chunk: str) -> ContextualChunk:
        prompt = CONTEXTUAL_PROMPT.format(
            WHOLE_DOCUMENT=doc,
            CHUNK_CONTENT=chunk
        )
        context = await self.llm_client.generate(
            prompt,
            max_tokens=100,
            cache_control={"type": "ephemeral"}  # Cache document
        )
        return ContextualChunk(
            chunk=chunk,
            context=context,
            full_text=f"{context}\n\n{chunk}"
        )
```

#### Phase 2: Hybrid Search (Week 3)
```python
class HybridRetriever:
    def __init__(self, vector_db, bm25_index):
        self.vector_db = vector_db
        self.bm25_index = bm25_index

    async def retrieve(self, query: str, top_k=20) -> List[Chunk]:
        # Parallel search
        vector_task = self.vector_search(query, k=150)
        bm25_task = self.bm25_search(query, k=150)

        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task
        )

        # Fuse rankings
        fused = reciprocal_rank_fusion(
            vector_results, bm25_results, k=60
        )

        return fused[:top_k]
```

#### Phase 3: Reranking (Week 4, Optional)
```python
class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Chunk], top_k=20):
        scores = self.model.predict([
            (query, chunk.full_text) for chunk in candidates
        ])
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [chunk for chunk, _ in ranked[:top_k]]
```

### Configuration and Deployment

#### Configuration Schema
```python
from pydantic import BaseModel, Field

class ContextualRAGConfig(BaseModel):
    """Configuration for Contextual RAG."""

    # Chunking parameters
    chunk_size: int = Field(800, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0, le=500)

    # Context generation
    context_max_tokens: int = Field(100, ge=50, le=200)
    context_model: str = "llama3.2:3b"  # Local Ollama model
    use_prompt_caching: bool = True

    # Retrieval parameters
    enable_hybrid_search: bool = True
    enable_reranking: bool = False
    top_k_retrieval: int = Field(20, ge=1, le=100)
    rrf_k_constant: int = Field(60, ge=10, le=100)

    # Collection naming
    collection_suffix: str = "_contextual"
```

#### ChromaDB Collection Strategy
```python
# Separate collections for traditional vs contextual RAG
traditional_collection = client.get_or_create_collection(
    name="documents_traditional",
    embedding_function=embedding_model
)

contextual_collection = client.get_or_create_collection(
    name="documents_contextual",
    embedding_function=embedding_model,
    metadata={"type": "contextual_rag", "chunk_size": 800}
)
```

### Testing and Validation

#### Metrics to Track
```python
@dataclass
class RAGMetrics:
    """Track RAG performance metrics."""
    retrieval_failure_rate: float  # Target: <3%
    top_k_precision: float  # Relevant chunks in top-k
    top_k_recall: float  # How many relevant chunks retrieved
    average_latency_ms: float  # End-to-end retrieval time
    cost_per_query: float  # $ cost for context generation
```

#### A/B Testing Framework
```python
async def compare_rag_approaches(queries: List[str]):
    """Compare traditional vs contextual RAG."""
    results = {
        "traditional": [],
        "contextual": []
    }

    for query in queries:
        # Traditional RAG
        trad_results = await traditional_rag.retrieve(query)
        results["traditional"].append(evaluate(query, trad_results))

        # Contextual RAG
        ctx_results = await contextual_rag.retrieve(query)
        results["contextual"].append(evaluate(query, ctx_results))

    return compare_metrics(results)
```

**Implementation Notes**:

**For OpenRAG MCP Server**:
1. Add new tool: `ingest_with_context(document, use_contextual=True)`
2. Store both traditional and contextual embeddings in separate collections
3. Use Ollama with llama3.2:3b for local context generation (no API costs)
4. Implement prompt caching at application level (cache document in memory)
5. Default to 800-token chunks for contextual mode
6. Make hybrid search configurable (can disable for faster queries)
7. Add metrics endpoint to track retrieval quality
8. Store context generation in metadata for debugging

**Performance Expectations**:
- Context generation: ~1-2 seconds per chunk with Ollama
- For 100-chunk document: ~2-3 minutes total (with async processing)
- Retrieval latency: +10-20ms vs traditional (hybrid search overhead)
- Quality improvement: 35-67% reduction in retrieval failures

**Cost Considerations**:
- Using local Ollama: Zero ongoing costs
- Storage overhead: ~15-20% increase (context adds 50-100 tokens/chunk)
- Compute overhead: One-time during ingestion
- Trade-off: Upfront processing time vs. better retrieval quality

**Future Enhancements**:
- Implement prompt caching for Ollama (if supported)
- Add domain-specific context prompts
- Support incremental context updates
- Implement context quality scoring
- Add automatic fallback to traditional RAG if context generation fails

**Tags**: #contextual-rag #anthropic #chunk-augmentation #hybrid-search #reciprocal-rank-fusion #async-processing #prompt-caching

---

## 2025-11-12 - Python Asyncio Background Tasks and Fire-and-Forget Patterns

**Context**: Research for implementing async background processing in OpenRAG MCP server, specifically for fire-and-forget operations like document ingestion and context generation. Proper async patterns prevent task garbage collection and ensure reliable background execution.

**Sources**:
- Python Official Docs: https://docs.python.org/3/library/asyncio-task.html
- Better Programming Guide: https://betterprogramming.pub/solve-common-asynchronous-scenarios-fire-and-forget-pub-sub-and-data-pipelines-with-python-asyncio-7f20d1268ade
- Stack Overflow Discussion: https://stackoverflow.com/questions/37278647/fire-and-forget-python-async-await
- Python Tutorials Deep Dive: https://www.pythontutorials.net/blog/python-asyncio-create-task-really-need-to-keep-a-reference/
- Super Fast Python: https://superfastpython.com/asyncio-disappearing-task-bug/
- MCP Background Processing: https://www.arsturn.com/blog/no-more-timeouts-how-to-build-long-running-mcp-tools-that-actually-finish-the-job
- GitHub Issue #104091: https://github.com/python/cpython/issues/104091

**Key Findings**:

### The Fundamental Problem

**Event Loop Weak References**:
- The asyncio event loop only keeps **weak references** to tasks
- Tasks without external references can be garbage collected at any time
- This can happen even while the task is still running
- Results in "disappearing tasks" that never complete

**Example of the Bug**:
```python
async def background_work():
    await asyncio.sleep(5)
    print("This may never print!")

async def main():
    asyncio.create_task(background_work())  # No reference kept!
    await asyncio.sleep(0.1)  # May get garbage collected here
    # background_work() might not complete
```

### The Set + Discard Pattern (RECOMMENDED)

**Official Python Documentation Solution**:
```python
# Module-level or class-level set to track background tasks
background_tasks = set()

async def fire_and_forget(coro):
    """
    Execute a coroutine in the background without awaiting.

    Args:
        coro: Coroutine to execute
    """
    task = asyncio.create_task(coro)

    # Add to set to prevent garbage collection
    background_tasks.add(task)

    # Remove from set when done to prevent memory leak
    task.add_done_callback(background_tasks.discard)

# Usage
await fire_and_forget(background_work())
```

**Why This Works**:
1. **Strong Reference**: Adding task to set prevents garbage collection
2. **Automatic Cleanup**: `add_done_callback(set.discard)` removes task when complete
3. **Memory Efficient**: Finished tasks don't accumulate in the set
4. **Exception Safe**: Cleanup happens regardless of success/failure

### Alternative Patterns

#### 1. TaskGroup (Python 3.11+)

**Modern Structured Concurrency**:
```python
async def main():
    async with asyncio.TaskGroup() as group:
        # Tasks are automatically tracked and awaited
        group.create_task(background_work_1())
        group.create_task(background_work_2())

        # Can continue with other work
        await other_work()

    # All tasks guaranteed to complete before exiting context
```

**Benefits**:
- Automatic exception handling
- All tasks guaranteed to complete or be cancelled
- Cleaner error propagation
- Prevents orphaned tasks

**Limitations**:
- Python 3.11+ only
- Not true "fire-and-forget" (waits at context exit)
- Better for structured concurrency than background tasks

#### 2. Explicit Task Tracking

**Manual Management**:
```python
class TaskTracker:
    def __init__(self):
        self.tasks: Set[asyncio.Task] = set()

    def create_task(self, coro) -> asyncio.Task:
        """Create and track a background task."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def wait_all(self, timeout: Optional[float] = None):
        """Wait for all tracked tasks to complete."""
        if not self.tasks:
            return
        await asyncio.wait(self.tasks, timeout=timeout)

    def cancel_all(self):
        """Cancel all tracked tasks."""
        for task in self.tasks:
            task.cancel()

# Usage
tracker = TaskTracker()
tracker.create_task(background_work())
```

#### 3. Gather for Concurrent Processing

**Batch Processing**:
```python
async def process_batch(items: List[Any]):
    """Process multiple items concurrently and wait for all."""
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle results
    for item, result in zip(items, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {item}: {result}")
        else:
            logger.info(f"Processed {item}: {result}")

    return results
```

**When to Use**:
- Need to wait for all tasks to complete
- Want to collect results from all tasks
- Need centralized error handling
- Not true fire-and-forget, but coordinated parallel execution

### Exception Handling in Background Tasks

**The Silent Failure Problem**:
```python
async def failing_task():
    await asyncio.sleep(1)
    raise ValueError("This error might never be seen!")

# Without proper handling, exception is lost
asyncio.create_task(failing_task())
```

**Proper Exception Handling**:
```python
async def safe_background_task(coro, error_handler=None):
    """Wrapper to handle exceptions in background tasks."""
    try:
        await coro
    except Exception as e:
        if error_handler:
            error_handler(e)
        else:
            logger.exception(f"Background task failed: {e}")
        # Optionally re-raise or return error signal

# Usage
background_tasks = set()

async def fire_and_forget_safe(coro, error_handler=None):
    task = asyncio.create_task(safe_background_task(coro, error_handler))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

**Checking Task Exceptions**:
```python
def check_task_exception(task: asyncio.Task):
    """Callback to check for exceptions after task completes."""
    try:
        # Calling exception() retrieves exception if task failed
        exception = task.exception()
        if exception:
            logger.error(f"Task failed with: {exception}")
    except asyncio.CancelledError:
        logger.warning("Task was cancelled")

# Add exception checker along with discard
task.add_done_callback(check_task_exception)
task.add_done_callback(background_tasks.discard)
```

### MCP Server Specific Patterns

#### Pattern 1: Asynchronous Hand-Off

**Long-Running Operations**:
```python
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types

# Track background tasks
background_tasks = set()

@server.call_tool()
async def ingest_document(
    arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Ingest document with background processing."""
    document_path = arguments["path"]

    # Generate job ID for tracking
    job_id = str(uuid.uuid4())

    # Start background ingestion
    task = asyncio.create_task(
        process_document_async(job_id, document_path)
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    # Return immediately with job ID
    return [types.TextContent(
        type="text",
        text=f"Document ingestion started. Job ID: {job_id}"
    )]

async def process_document_async(job_id: str, path: str):
    """Background processing of document."""
    try:
        # Long-running work
        document = await load_document(path)
        chunks = await chunk_document(document)
        await generate_contexts(chunks)
        await store_embeddings(chunks)

        # Update job status
        job_store[job_id] = {"status": "complete", "chunks": len(chunks)}
    except Exception as e:
        logger.exception(f"Document processing failed: {e}")
        job_store[job_id] = {"status": "failed", "error": str(e)}
```

#### Pattern 2: Status Polling

**Query Background Job Status**:
```python
@server.call_tool()
async def query_job_status(
    arguments: dict
) -> list[types.TextContent]:
    """Check status of background job."""
    job_id = arguments["job_id"]

    status = job_store.get(job_id, {"status": "not_found"})

    return [types.TextContent(
        type="text",
        text=json.dumps(status, indent=2)
    )]
```

#### Pattern 3: Progress Streaming

**Stream Progress Updates**:
```python
async def process_with_progress(job_id: str, items: List[Any]):
    """Process items with progress tracking."""
    total = len(items)

    for idx, item in enumerate(items, 1):
        await process_item(item)

        # Update progress
        progress = (idx / total) * 100
        job_store[job_id] = {
            "status": "processing",
            "progress": progress,
            "completed": idx,
            "total": total
        }
```

### Performance Considerations

#### Concurrency Limits

**Avoid Overwhelming the System**:
```python
from asyncio import Semaphore

class BoundedTaskManager:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = Semaphore(max_concurrent)
        self.tasks = set()

    async def create_task(self, coro):
        """Create task with concurrency limit."""
        async def bounded_coro():
            async with self.semaphore:
                return await coro

        task = asyncio.create_task(bounded_coro())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

# Usage: Limit to 10 concurrent document ingestions
task_manager = BoundedTaskManager(max_concurrent=10)
await task_manager.create_task(ingest_document(path))
```

#### Timeout Handling

**Prevent Infinite Execution**:
```python
async def task_with_timeout(coro, timeout: float):
    """Execute task with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Task timed out after {timeout}s")
        raise
```

### Best Practices Summary

#### DO:
1. **Always keep strong references** to background tasks (use set pattern)
2. **Add done callbacks** to clean up finished tasks (`set.discard`)
3. **Wrap tasks in exception handlers** to prevent silent failures
4. **Log all background task failures** for debugging
5. **Use TaskGroup for structured concurrency** (Python 3.11+)
6. **Limit concurrency** with semaphores for resource-intensive tasks
7. **Provide status endpoints** for long-running operations
8. **Implement timeouts** to prevent infinite execution

#### DON'T:
1. **Don't create tasks without keeping references** (causes disappearing tasks)
2. **Don't ignore exceptions** in background tasks
3. **Don't create unlimited concurrent tasks** (can overwhelm system)
4. **Don't use fire-and-forget for critical operations** without status tracking
5. **Don't assume background tasks will complete** if main task exits early
6. **Don't forget to clean up** finished tasks (causes memory leaks)

### Implementation for OpenRAG

**Core Task Manager**:
```python
from typing import Optional, Callable
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TaskInfo:
    """Information about a background task."""
    task_id: str
    created_at: datetime
    description: str
    status: str = "running"
    error: Optional[str] = None

class OpenRAGTaskManager:
    """Manage background tasks for OpenRAG MCP server."""

    def __init__(self, max_concurrent: int = 10):
        self.tasks: Set[asyncio.Task] = set()
        self.task_info: Dict[str, TaskInfo] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def create_task(
        self,
        coro,
        task_id: str,
        description: str,
        error_handler: Optional[Callable] = None
    ) -> str:
        """
        Create and track a background task.

        Args:
            coro: Coroutine to execute
            task_id: Unique identifier for task
            description: Human-readable description
            error_handler: Optional callback for errors

        Returns:
            task_id for status tracking
        """
        # Store task info
        self.task_info[task_id] = TaskInfo(
            task_id=task_id,
            created_at=datetime.now(),
            description=description
        )

        # Wrap coroutine with error handling and semaphore
        async def managed_coro():
            async with self.semaphore:
                try:
                    result = await coro
                    self.task_info[task_id].status = "completed"
                    return result
                except Exception as e:
                    logger.exception(f"Task {task_id} failed: {e}")
                    self.task_info[task_id].status = "failed"
                    self.task_info[task_id].error = str(e)

                    if error_handler:
                        error_handler(e)

                    raise

        # Create task with cleanup
        task = asyncio.create_task(managed_coro())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        return task_id

    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get status of a background task."""
        return self.task_info.get(task_id)

    async def wait_all(self, timeout: Optional[float] = None):
        """Wait for all tasks to complete."""
        if not self.tasks:
            return
        await asyncio.wait(self.tasks, timeout=timeout)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        for task in self.tasks:
            if task.get_name() == task_id:
                task.cancel()
                self.task_info[task_id].status = "cancelled"
                return True
        return False

# Global instance
task_manager = OpenRAGTaskManager(max_concurrent=5)
```

**Usage in MCP Tools**:
```python
@server.call_tool()
async def ingest_with_context(
    arguments: dict
) -> list[types.TextContent]:
    """Ingest document with contextual embedding (background)."""
    document_path = arguments["path"]
    job_id = str(uuid.uuid4())

    # Start background processing
    task_manager.create_task(
        coro=ingest_document_contextual(document_path),
        task_id=job_id,
        description=f"Contextual ingestion: {document_path}"
    )

    return [types.TextContent(
        type="text",
        text=f"Started contextual ingestion. Job ID: {job_id}\n"
             f"Use query_job_status with this ID to check progress."
    )]

@server.call_tool()
async def query_job_status(
    arguments: dict
) -> list[types.TextContent]:
    """Query status of background job."""
    job_id = arguments["job_id"]

    status = task_manager.get_status(job_id)

    if not status:
        return [types.TextContent(
            type="text",
            text=f"Job {job_id} not found."
        )]

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "job_id": status.task_id,
            "status": status.status,
            "description": status.description,
            "created_at": status.created_at.isoformat(),
            "error": status.error
        }, indent=2)
    )]
```

**Implementation Notes**:

**For OpenRAG MCP Server**:
1. Use set + discard pattern for all background tasks
2. Implement TaskManager class for centralized task tracking
3. Limit concurrent document ingestion (5-10 concurrent max)
4. Provide status query endpoint for all async operations
5. Log all task failures to stderr (MCP requirement)
6. Store task metadata (job ID, status, error) for debugging
7. Implement graceful shutdown (wait for tasks or cancel)
8. Use semaphores to prevent resource exhaustion

**Gotchas to Avoid**:
- Never create task without adding to tracking set
- Always add done callback before any await that might fail
- Clean up task_info dict periodically (prevent memory leak)
- Don't forget semaphore for resource-intensive operations
- Log exceptions in background tasks (they're otherwise silent)
- Test with asyncio debug mode enabled during development

**Tags**: #asyncio #background-tasks #fire-and-forget #task-management #MCP-server #concurrency #semaphore #exception-handling

---

## 2025-11-12 - ChromaDB Multi-Collection Architecture

**Context**: Research for organizing multiple ChromaDB collections in OpenRAG to separate traditional RAG and contextual RAG embeddings, plus support for different document types. Proper collection architecture enables flexibility and clear separation of concerns.

**Sources**:
- ChromaDB Cookbook - Collections: https://cookbook.chromadb.dev/core/collections/
- ChromaDB Docs - Manage Collections: https://docs.trychroma.com/docs/collections/manage-collections
- ChromaDB Cookbook - Concepts: https://cookbook.chromadb.dev/core/concepts/
- Stack Overflow - Combining Databases: https://stackoverflow.com/questions/76048941/how-to-combine-two-chroma-databases
- LangChain Discussion #5849: https://github.com/langchain-ai/langchain/discussions/5849
- Private GPT Discussion #298: https://github.com/zylon-ai/private-gpt/discussions/298
- DataCamp Tutorial: https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
- Analytics Vidhya Guide: https://www.analyticsvidhya.com/blog/2023/07/guide-to-chroma-db-a-vector-store-for-your-generative-ai-llms/

**Key Findings**:

### ChromaDB Organizational Hierarchy

**Three-Level Structure**:
```
Tenant (Organization/User)
├── Database (Application/Project)
│   ├── Collection 1 (Document Set)
│   ├── Collection 2 (Document Set)
│   └── Collection N (Document Set)
```

**Default Setup**:
```python
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE

client = chromadb.PersistentClient(
    path="./chroma_db",
    tenant=DEFAULT_TENANT,  # "default_tenant"
    database=DEFAULT_DATABASE  # "default_database"
)
```

### Collection Management Operations

#### 1. Creating Collections

**Basic Creation**:
```python
# Simple creation
collection = client.create_collection(name="my_documents")

# With metadata and embedding function (RECOMMENDED)
collection = client.create_collection(
    name="contextual_embeddings",
    metadata={
        "description": "Contextual RAG embeddings",
        "type": "contextual",
        "chunk_size": 800,
        "hnsw:space": "cosine"  # Distance metric
    },
    embedding_function=embedding_model
)
```

**Get or Create (Idempotent)**:
```python
# Safer for production - won't fail if exists
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_model
)
```

#### 2. Retrieving Collections

**List All Collections**:
```python
# Get all collections (paginated)
collections = client.list_collections(
    limit=100,  # Max 100 per request
    offset=0     # For pagination
)

for collection in collections:
    print(f"{collection.name}: {collection.count()} documents")
```

**Get Specific Collection**:
```python
try:
    collection = client.get_collection(name="my_documents")
except Exception as e:
    print(f"Collection not found: {e}")
```

#### 3. Modifying Collections

**Update Metadata or Name**:
```python
collection.modify(
    name="updated_collection_name",  # Optional
    metadata={"new_key": "new_value"}  # Merges with existing
)
```

**Important**: Cannot change distance metric after creation!

#### 4. Deleting Collections

**Permanent Deletion**:
```python
client.delete_collection(name="my_documents")
# WARNING: Destructive and not reversible!
```

### Collection Configuration Options

#### Distance Metrics

**Available Options** (set at creation, immutable):
```python
metadata = {
    "hnsw:space": "cosine"  # Options: "l2", "ip", "cosine"
}

# "l2": Euclidean distance (L2 norm)
# "ip": Inner product
# "cosine": Cosine similarity (RECOMMENDED for text)
```

**Choosing Distance Metric**:
- **Cosine**: Best for text embeddings (default for most models)
- **L2**: Good for normalized vectors
- **IP**: Useful for specific similarity tasks

#### HNSW Index Configuration

**Performance Tuning**:
```python
metadata = {
    "hnsw:batch_size": 100,  # Batch size for HNSW index (default: 100)
    "hnsw:sync_threshold": 1000,  # Sync to disk after N adds
    "hnsw:M": 16,  # Max connections per node (affects recall/speed)
    "hnsw:ef_construction": 200,  # Construction time effort
    "hnsw:ef_search": 100  # Query time effort
}
```

**Tuning Guidelines**:
- **hnsw:batch_size**: Increase for bulk ingestion (100-1000)
- **hnsw:M**: Higher = better recall, more memory (8-64)
- **hnsw:ef_construction**: Higher = better index quality, slower (100-500)
- **hnsw:ef_search**: Higher = better recall, slower queries (10-500)

### Multi-Collection Architecture Patterns

#### Pattern 1: Separation by RAG Type

**Traditional vs Contextual RAG**:
```python
class MultiRAGCollections:
    def __init__(self, client: chromadb.PersistentClient):
        self.client = client

        # Traditional RAG collection
        self.traditional = client.get_or_create_collection(
            name="documents_traditional",
            metadata={
                "type": "traditional_rag",
                "chunk_size": 400,
                "overlap": 60,
                "hnsw:space": "cosine"
            },
            embedding_function=embedding_model
        )

        # Contextual RAG collection
        self.contextual = client.get_or_create_collection(
            name="documents_contextual",
            metadata={
                "type": "contextual_rag",
                "chunk_size": 800,
                "overlap": 100,
                "has_context": True,
                "hnsw:space": "cosine"
            },
            embedding_function=embedding_model
        )

    async def ingest_document(self, doc: str, use_contextual: bool = True):
        """Ingest to appropriate collection based on mode."""
        if use_contextual:
            chunks = await create_contextual_chunks(doc)
            self.contextual.add(
                documents=[c.full_text for c in chunks],
                metadatas=[c.metadata for c in chunks],
                ids=[c.id for c in chunks]
            )
        else:
            chunks = create_traditional_chunks(doc)
            self.traditional.add(
                documents=chunks,
                ids=[str(uuid.uuid4()) for _ in chunks]
            )
```

#### Pattern 2: Separation by Document Type

**Type-Specific Collections**:
```python
class DocumentTypeCollections:
    """Separate collections for different document types."""

    def __init__(self, client: chromadb.PersistentClient):
        self.client = client
        self.collections = {}

        # Define document types
        doc_types = ["pdf", "markdown", "code", "web", "email"]

        for doc_type in doc_types:
            self.collections[doc_type] = client.get_or_create_collection(
                name=f"documents_{doc_type}",
                metadata={
                    "document_type": doc_type,
                    "hnsw:space": "cosine"
                },
                embedding_function=embedding_model
            )

    def get_collection(self, doc_type: str):
        """Get collection for specific document type."""
        return self.collections.get(doc_type)

    async def ingest_document(self, doc: str, doc_type: str):
        """Ingest to type-specific collection."""
        collection = self.get_collection(doc_type)
        if not collection:
            raise ValueError(f"Unknown document type: {doc_type}")

        chunks = await process_document(doc, doc_type)
        collection.add(
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[c.id for c in chunks]
        )
```

#### Pattern 3: Hybrid Metadata + Multi-Collection

**Flexible Organization**:
```python
class HybridCollectionStrategy:
    """Combine collections with metadata filtering."""

    def __init__(self, client: chromadb.PersistentClient):
        self.client = client

        # One collection per RAG type
        self.traditional = client.get_or_create_collection("traditional")
        self.contextual = client.get_or_create_collection("contextual")

    async def ingest_document(
        self,
        doc: str,
        doc_type: str,
        tags: List[str],
        use_contextual: bool = True
    ):
        """Ingest with rich metadata."""
        collection = self.contextual if use_contextual else self.traditional
        chunks = await process_document(doc, use_contextual)

        collection.add(
            documents=[c.text for c in chunks],
            metadatas=[{
                "doc_type": doc_type,
                "tags": tags,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "source": doc.name,
                "ingested_at": datetime.now().isoformat()
            } for idx, c in enumerate(chunks)],
            ids=[c.id for c in chunks]
        )

    async def query_by_type(
        self,
        query: str,
        doc_type: str,
        use_contextual: bool = True
    ):
        """Query specific document type."""
        collection = self.contextual if use_contextual else self.traditional

        results = collection.query(
            query_texts=[query],
            n_results=20,
            where={"doc_type": doc_type}  # Metadata filter
        )

        return results
```

### Collection Naming Conventions

**Validation Rules**:
- Length: 3-63 characters
- Start/end: Alphanumeric
- Contents: Alphanumeric, underscore, hyphen
- No consecutive periods (..)
- Not a valid IPv4 address

**Recommended Naming Scheme**:
```python
# Format: {content_type}_{rag_type}_{version}
COLLECTION_NAMES = {
    "traditional": "documents_traditional_v1",
    "contextual": "documents_contextual_v1",
    "pdf_traditional": "pdf_traditional_v1",
    "pdf_contextual": "pdf_contextual_v1",
    "code_traditional": "code_traditional_v1",
    "code_contextual": "code_contextual_v1"
}

def get_collection_name(content_type: str, rag_type: str, version: int = 1):
    """Generate standard collection name."""
    return f"{content_type}_{rag_type}_v{version}"
```

### Query Strategies Across Collections

#### Strategy 1: Query Multiple Collections

**Parallel Queries**:
```python
async def query_all_collections(query: str, top_k: int = 20):
    """Query all collections and combine results."""

    # Query in parallel
    traditional_task = asyncio.create_task(
        query_collection(traditional_collection, query, top_k)
    )
    contextual_task = asyncio.create_task(
        query_collection(contextual_collection, query, top_k)
    )

    traditional_results, contextual_results = await asyncio.gather(
        traditional_task, contextual_task
    )

    # Combine and deduplicate
    combined = combine_results(traditional_results, contextual_results)
    return combined[:top_k]
```

#### Strategy 2: Smart Collection Selection

**Query-Aware Routing**:
```python
def select_collection(query: str, context: dict) -> str:
    """Select best collection based on query and context."""

    # Check user preference
    if context.get("prefer_contextual"):
        return "contextual"

    # Check query complexity
    if len(query.split()) > 10:  # Complex query
        return "contextual"  # Better context

    # Check document type preference
    doc_type = context.get("document_type")
    if doc_type:
        return f"{doc_type}_contextual"

    # Default to contextual
    return "contextual"
```

### Performance Optimization

#### Batch Operations

**Efficient Bulk Inserts**:
```python
async def batch_ingest(documents: List[Document], batch_size: int = 100):
    """Ingest documents in batches for efficiency."""

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        collection.add(
            documents=[d.text for d in batch],
            metadatas=[d.metadata for d in batch],
            ids=[d.id for d in batch]
        )

        # Optional: Add delay to prevent overwhelming
        await asyncio.sleep(0.1)
```

#### Collection Statistics

**Monitor Performance**:
```python
def get_collection_stats(collection) -> dict:
    """Get statistics about a collection."""

    count = collection.count()
    peek = collection.peek(limit=10)

    return {
        "name": collection.name,
        "count": count,
        "metadata": collection.metadata,
        "sample_docs": len(peek["documents"]),
        "has_embeddings": len(peek.get("embeddings", [])) > 0
    }

# Usage
for collection in client.list_collections():
    stats = get_collection_stats(collection)
    print(f"{stats['name']}: {stats['count']} documents")
```

### Migration and Maintenance

#### Reindexing Strategy

**Copy Collection with Different Settings**:
```python
async def reindex_collection(
    source_name: str,
    target_name: str,
    new_embedding_func=None,
    batch_size: int = 100
):
    """Reindex collection with new settings."""

    # Get source collection
    source = client.get_collection(source_name)

    # Create target with new settings
    target = client.create_collection(
        name=target_name,
        embedding_function=new_embedding_func or embedding_model,
        metadata={"reindexed_from": source_name}
    )

    # Get all documents in batches
    total = source.count()
    for offset in range(0, total, batch_size):
        batch = source.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )

        # Add to target
        target.add(
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            ids=batch["ids"]
        )

        logger.info(f"Reindexed {offset + len(batch['ids'])}/{total}")
```

#### Backup Strategy

**Export Collection Data**:
```python
import json

def backup_collection(collection, output_path: str):
    """Backup collection to JSON file."""

    # Get all data
    data = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    backup = {
        "name": collection.name,
        "metadata": collection.metadata,
        "count": collection.count(),
        "data": {
            "ids": data["ids"],
            "documents": data["documents"],
            "metadatas": data["metadatas"],
            "embeddings": [[float(x) for x in emb] for emb in data["embeddings"]]
        }
    }

    with open(output_path, 'w') as f:
        json.dump(backup, f, indent=2)

def restore_collection(client, backup_path: str):
    """Restore collection from backup."""

    with open(backup_path, 'r') as f:
        backup = json.load(f)

    # Create collection
    collection = client.create_collection(
        name=backup["name"],
        metadata=backup["metadata"]
    )

    # Restore data
    collection.add(
        ids=backup["data"]["ids"],
        documents=backup["data"]["documents"],
        metadatas=backup["data"]["metadatas"],
        embeddings=backup["data"]["embeddings"]
    )
```

### Implementation for OpenRAG

**Recommended Architecture**:
```python
class OpenRAGCollections:
    """Manage all ChromaDB collections for OpenRAG."""

    def __init__(self, client: chromadb.PersistentClient):
        self.client = client
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

        # Core collections
        self.collections = {
            "traditional": self._create_collection("documents_traditional_v1", {
                "type": "traditional_rag",
                "chunk_size": 400,
                "overlap": 60
            }),
            "contextual": self._create_collection("documents_contextual_v1", {
                "type": "contextual_rag",
                "chunk_size": 800,
                "overlap": 100,
                "has_context": True
            })
        }

    def _create_collection(self, name: str, metadata: dict):
        """Create collection with standard settings."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={
                **metadata,
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:ef_construction": 200
            },
            embedding_function=self.embedding_model
        )

    def get_collection(self, rag_type: str = "contextual"):
        """Get collection by RAG type."""
        return self.collections.get(rag_type)

    def list_all_collections(self) -> List[dict]:
        """Get statistics for all collections."""
        return [
            {
                "name": name,
                "count": coll.count(),
                "metadata": coll.metadata
            }
            for name, coll in self.collections.items()
        ]
```

**Implementation Notes**:

**For OpenRAG MCP Server**:
1. Start with two collections: traditional and contextual
2. Use descriptive names with version suffix (_v1)
3. Store RAG type and parameters in collection metadata
4. Default to cosine distance for text embeddings
5. Implement collection selection logic in query tools
6. Provide tool to list all collections and their stats
7. Support both single-collection and multi-collection queries
8. Implement backup/restore tools for collections
9. Add collection migration tool for reindexing
10. Monitor collection sizes and performance metrics

**Configuration**:
```python
class CollectionConfig(BaseModel):
    """Configuration for ChromaDB collections."""

    # Collection naming
    prefix: str = "documents"
    version: int = 1

    # Storage
    chroma_path: str = "./chroma_db"

    # HNSW settings
    hnsw_space: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_batch_size: int = 100

    # Management
    enable_backup: bool = True
    backup_interval_hours: int = 24
    max_collection_size: int = 1_000_000  # Max documents per collection
```

**Tags**: #chromadb #multi-collection #collection-management #vector-database #metadata #HNSW #performance #backup #migration

---

## 2025-11-12 - Ollama Integration for Document Summarization and Context Generation

**Context**: Research for using Ollama with local models (especially llama3.2) for generating contextual embeddings and document summaries in OpenRAG. Focus on error handling, timeout management, and best practices for production deployment.

**Sources**:
- Ollama Official - llama3.2: https://ollama.com/library/llama3.2
- Ollama Official - llama3.1: https://ollama.com/library/llama3.1
- Collabnix Production Guide: https://collabnix.com/ollama-api-integration-building-production-ready-llm-applications/
- MarkAI Code Integration Guide: https://markaicode.com/python-ollama-integration-sdk-guide/
- Medium - Exploring llama3.2: https://medium.com/@kingnathanal/ollama-drama-exploring-ollama-with-llama-3-2-genai-and-hosting-llm-models-locally-df7b440c98e1
- DZone Document Summarization: https://dzone.com/articles/build-a-local-ai-powered-document-summarization-tool
- GitHub Timeout Issue #2424: https://github.com/ollama/ollama/issues/2424
- Medium Timeout Solution: https://medium.com/@loic.dl.nlp/i-solved-the-timeout-problem-as-follows-6c06e421439a
- Stack Overflow Summarization: https://stackoverflow.com/questions/78855296/text-summarization-with-llm-llama-3-1-8b-instruct

**Key Findings**:

### Llama 3.2 Model Capabilities

**Text Models Available**:
- **1B model**: Ultra-lightweight, 128K context, optimized for edge devices
- **3B model**: Sweet spot for local deployment, 128K context
- **11B vision model**: Multimodal (text + images)
- **90B vision model**: Large multimodal model

**Key Capabilities for RAG**:
- Instruction following
- Document summarization
- Prompt rewriting
- Context generation
- 128K token context window (all models)
- Multilingual support

**Performance Benchmarks**:
- 3B model outperforms Gemma 2 2.6B and Phi 3.5-mini on summarization tasks
- State-of-the-art in size class for on-device use cases
- Optimized for agentic retrieval and summarization

### Ollama Python Integration

#### Basic Setup

**Installation**:
```bash
# Install Ollama (macOS)
brew install ollama

# Install Python client
pip install ollama

# Pull model
ollama pull llama3.2:3b
```

**Basic Generation**:
```python
import ollama

def generate_text(prompt: str, model: str = "llama3.2:3b") -> str:
    """Generate text using Ollama."""
    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False
    )
    return response['response']

# Usage
summary = generate_text("Summarize this document: ...")
```

#### Streaming Responses

**For Real-Time Feedback**:
```python
def stream_response(prompt: str, model: str = "llama3.2:3b"):
    """Stream response token by token."""
    stream = ollama.generate(
        model=model,
        prompt=prompt,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if 'response' in chunk:
            token = chunk['response']
            print(token, end='', flush=True)
            full_response += token

    return full_response
```

### Timeout and Error Handling

#### The Timeout Problem

**Common Issues**:
- Default timeout too short for large documents
- Model loading can take 30+ seconds first time
- Context generation for 100+ chunks hits timeout
- Network issues between client and Ollama server

#### Solution 1: Configure Timeout

**Client-Level Timeout**:
```python
import ollama
from ollama import Client

# Method 1: Custom client with timeout
client = Client(
    host='http://localhost:11434',
    timeout=300.0  # 5 minutes
)

response = client.generate(
    model="llama3.2:3b",
    prompt=prompt
)

# Method 2: Using httpx directly
import httpx

http_client = httpx.Client(timeout=300.0)
client = Client(httpx_client=http_client)
```

**Request-Level Timeout**:
```python
# Set per-request options
response = ollama.generate(
    model="llama3.2:3b",
    prompt=prompt,
    options={
        "num_predict": 2000,  # Max tokens to generate
        "temperature": 0.7,
        "top_p": 0.9
    }
)
```

#### Solution 2: Retry Logic with Exponential Backoff

**Robust Client**:
```python
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RobustOllamaClient:
    """Production-ready Ollama client with retry logic."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = 300.0
    ):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Configure client with timeout
        self.client = Client(
            host='http://localhost:11434',
            timeout=timeout
        )

    def generate_with_retry(
        self,
        prompt: str,
        **options
    ) -> Optional[str]:
        """Generate text with automatic retry on failure."""

        for attempt in range(self.max_retries):
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    options=options
                )
                return response['response']

            except Exception as e:
                logger.warning(
                    f"Generation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retries exhausted for prompt: {prompt[:100]}...")
                    raise

        return None

# Usage
client = RobustOllamaClient(timeout=300.0)
context = client.generate_with_retry(
    prompt=CONTEXTUAL_PROMPT.format(doc=document, chunk=chunk),
    num_predict=100,
    temperature=0.5
)
```

#### Solution 3: Async Implementation

**Non-Blocking Context Generation**:
```python
import asyncio
import aiohttp
from typing import List, Dict

class AsyncOllamaClient:
    """Async Ollama client for concurrent operations."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def generate(
        self,
        prompt: str,
        **options
    ) -> str:
        """Generate text asynchronously."""

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }

            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                result = await response.json()
                return result['response']

    async def generate_batch(
        self,
        prompts: List[str],
        **options
    ) -> List[str]:
        """Generate for multiple prompts concurrently."""

        tasks = [
            self.generate(prompt, **options)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors
        clean_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate for prompt {idx}: {result}")
                clean_results.append(None)
            else:
                clean_results.append(result)

        return clean_results

# Usage
async def generate_all_contexts(document: str, chunks: List[str]):
    client = AsyncOllamaClient(timeout=300.0)

    prompts = [
        CONTEXTUAL_PROMPT.format(doc=document, chunk=chunk)
        for chunk in chunks
    ]

    contexts = await client.generate_batch(
        prompts,
        num_predict=100,
        temperature=0.5
    )

    return contexts
```

### Model Selection for Different Tasks

#### Context Generation (Lightweight)

**Recommended: llama3.2:1b or llama3.2:3b**
```python
CONTEXT_GENERATION_CONFIG = {
    "model": "llama3.2:3b",  # Fast, sufficient for context
    "num_predict": 100,       # Short context (50-100 tokens)
    "temperature": 0.5,       # Moderate creativity
    "top_p": 0.9,
    "stop": ["\n\n", "</context>"]  # Stop at natural boundary
}

def generate_context(document: str, chunk: str) -> str:
    """Generate succinct context for chunk."""
    prompt = f"""<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Provide a brief 1-2 sentence context explaining what this chunk is about
within the document. Be concise and focus on the main topic.

Context:"""

    return ollama_client.generate_with_retry(
        prompt=prompt,
        **CONTEXT_GENERATION_CONFIG
    )
```

#### Document Summarization (Medium)

**Recommended: llama3.2:3b or llama3.1:8b**
```python
SUMMARIZATION_CONFIG = {
    "model": "llama3.2:3b",  # Good balance
    "num_predict": 500,       # Longer summary
    "temperature": 0.7,
    "top_p": 0.95
}

def summarize_document(document: str, max_words: int = 200) -> str:
    """Generate document summary."""
    prompt = f"""Summarize the following document in approximately {max_words} words.
Focus on the main points, key takeaways, and important details.

Document:
{document}

Summary:"""

    return ollama_client.generate_with_retry(
        prompt=prompt,
        **SUMMARIZATION_CONFIG
    )
```

#### Complex Analysis (Heavy)

**Recommended: llama3.1:8b or llama3.1:70b (if resources allow)**
```python
ANALYSIS_CONFIG = {
    "model": "llama3.1:8b",  # Better reasoning
    "num_predict": 1000,
    "temperature": 0.8,
    "top_p": 0.95
}

def analyze_document(document: str) -> Dict[str, str]:
    """Perform complex document analysis."""
    prompt = f"""Analyze the following document and provide:
1. Main topic (1 sentence)
2. Key points (3-5 bullet points)
3. Entities mentioned (people, places, organizations)
4. Sentiment/tone
5. Document type/category

Document:
{document}

Analysis:"""

    response = ollama_client.generate_with_retry(
        prompt=prompt,
        **ANALYSIS_CONFIG
    )

    # Parse structured response
    return parse_analysis(response)
```

### Performance Optimization

#### 1. Model Warm-Up

**Pre-Load Model**:
```python
def warmup_model(model: str = "llama3.2:3b"):
    """Warm up model to avoid first-request delay."""
    logger.info(f"Warming up model: {model}")

    ollama.generate(
        model=model,
        prompt="Test",
        options={"num_predict": 1}
    )

    logger.info(f"Model {model} ready")

# Call during server startup
warmup_model()
```

#### 2. Batching Strategy

**Process in Optimal Batches**:
```python
async def process_chunks_in_batches(
    document: str,
    chunks: List[str],
    batch_size: int = 10
) -> List[str]:
    """Process chunks in batches to avoid overwhelming."""

    contexts = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        logger.info(f"Processing batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}")

        batch_contexts = await generate_batch_contexts(document, batch)
        contexts.extend(batch_contexts)

        # Optional: Brief pause between batches
        await asyncio.sleep(0.5)

    return contexts
```

#### 3. Caching Strategy

**Cache Frequently Used Contexts**:
```python
from functools import lru_cache
import hashlib

class CachedOllamaClient:
    """Ollama client with response caching."""

    def __init__(self, cache_size: int = 1000):
        self.client = RobustOllamaClient()
        self.cache = {}
        self.cache_size = cache_size

    def _hash_prompt(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def generate_cached(self, prompt: str, **options) -> str:
        """Generate with caching."""
        cache_key = self._hash_prompt(prompt)

        if cache_key in self.cache:
            logger.debug("Cache hit")
            return self.cache[cache_key]

        response = self.client.generate_with_retry(prompt, **options)

        # Add to cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[cache_key] = response
        return response
```

### Error Handling Best Practices

#### Common Error Types

**1. Connection Errors**:
```python
from ollama import ResponseError
import requests.exceptions

try:
    response = ollama.generate(model="llama3.2:3b", prompt=prompt)
except requests.exceptions.ConnectionError as e:
    logger.error("Ollama server not reachable. Is it running?")
    # Check: ollama serve
    raise RuntimeError("Ollama server not available") from e
```

**2. Model Not Found**:
```python
try:
    response = ollama.generate(model="llama3.2:3b", prompt=prompt)
except ResponseError as e:
    if "model not found" in str(e).lower():
        logger.error("Model not pulled. Run: ollama pull llama3.2:3b")
        # Auto-pull (optional)
        ollama.pull("llama3.2:3b")
        # Retry
        response = ollama.generate(model="llama3.2:3b", prompt=prompt)
    else:
        raise
```

**3. Timeout Errors**:
```python
import asyncio

try:
    response = await asyncio.wait_for(
        generate_async(prompt),
        timeout=300.0  # 5 minutes
    )
except asyncio.TimeoutError:
    logger.error(f"Generation timed out after 300s")
    # Fallback strategy
    response = fallback_generation(prompt)
```

#### Graceful Degradation

**Fallback Strategy**:
```python
class FallbackOllamaClient:
    """Client with fallback to simpler models."""

    def __init__(self):
        self.primary_model = "llama3.2:3b"
        self.fallback_model = "llama3.2:1b"
        self.client = RobustOllamaClient()

    def generate_with_fallback(self, prompt: str, **options) -> str:
        """Try primary model, fallback to simpler if fails."""
        try:
            return self.client.generate_with_retry(
                prompt,
                model=self.primary_model,
                **options
            )
        except Exception as e:
            logger.warning(f"Primary model failed: {e}. Trying fallback...")

            try:
                return self.client.generate_with_retry(
                    prompt,
                    model=self.fallback_model,
                    **options
                )
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                # Last resort: return empty or use heuristic
                return generate_heuristic_context(prompt)
```

### Monitoring and Metrics

**Track Ollama Performance**:
```python
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class GenerationMetrics:
    """Track generation performance."""
    model: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None

class MonitoredOllamaClient:
    """Ollama client with performance monitoring."""

    def __init__(self):
        self.client = RobustOllamaClient()
        self.metrics: List[GenerationMetrics] = []

    def generate_monitored(self, prompt: str, **options) -> str:
        """Generate with performance tracking."""
        start_time = time.time()
        error = None
        success = True

        try:
            response = self.client.generate_with_retry(prompt, **options)

            # Parse response metadata
            prompt_tokens = len(prompt.split())  # Rough estimate
            generated_tokens = len(response.split())

        except Exception as e:
            error = str(e)
            success = False
            response = None
            prompt_tokens = 0
            generated_tokens = 0

        finally:
            latency_ms = (time.time() - start_time) * 1000

            # Record metrics
            self.metrics.append(GenerationMetrics(
                model=self.client.model,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                success=success,
                error=error
            ))

        return response

    def get_stats(self) -> dict:
        """Get performance statistics."""
        if not self.metrics:
            return {}

        latencies = [m.latency_ms for m in self.metrics if m.success]
        success_rate = sum(1 for m in self.metrics if m.success) / len(self.metrics)

        return {
            "total_requests": len(self.metrics),
            "success_rate": success_rate,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if latencies else 0,
            "total_tokens_generated": sum(m.generated_tokens for m in self.metrics)
        }
```

### Implementation for OpenRAG

**Production-Ready Ollama Integration**:
```python
from typing import Optional, List
import asyncio
import logging

logger = logging.getLogger(__name__)

class OpenRAGOllamaClient:
    """Ollama client for OpenRAG MCP server."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        timeout: float = 300.0,
        max_retries: int = 3,
        enable_caching: bool = True
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize client
        self.client = Client(
            host='http://localhost:11434',
            timeout=timeout
        )

        # Optional caching
        self.cache = {} if enable_caching else None

        # Warm up model
        self._warmup()

    def _warmup(self):
        """Warm up model during initialization."""
        try:
            logger.info(f"Warming up model: {self.model}")
            self.client.generate(
                model=self.model,
                prompt="Test",
                options={"num_predict": 1}
            )
            logger.info("Model ready")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def generate_context(
        self,
        document: str,
        chunk: str
    ) -> Optional[str]:
        """Generate context for a chunk."""

        prompt = f"""<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

        # Check cache
        if self.cache is not None:
            cache_key = hash((document[:1000], chunk))  # Hash for lookup
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Generate with retry
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.generate,
                    model=self.model,
                    prompt=prompt,
                    options={
                        "num_predict": 100,
                        "temperature": 0.5,
                        "stop": ["\n\n", "Document:", "Chunk:"]
                    }
                )

                context = response['response'].strip()

                # Cache result
                if self.cache is not None:
                    self.cache[cache_key] = context

                return context

            except Exception as e:
                logger.warning(
                    f"Context generation failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("All retries exhausted")
                    return None

    async def generate_contexts_batch(
        self,
        document: str,
        chunks: List[str],
        batch_size: int = 10
    ) -> List[Optional[str]]:
        """Generate contexts for multiple chunks."""

        contexts = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            logger.info(
                f"Processing context batch {i//batch_size + 1}/"
                f"{(len(chunks) + batch_size - 1) // batch_size}"
            )

            # Process batch concurrently
            batch_contexts = await asyncio.gather(*[
                self.generate_context(document, chunk)
                for chunk in batch
            ])

            contexts.extend(batch_contexts)

            # Brief pause between batches
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.5)

        return contexts

# Initialize global client
ollama_client = OpenRAGOllamaClient(
    model="llama3.2:3b",
    timeout=300.0,
    enable_caching=True
)
```

**Implementation Notes**:

**For OpenRAG MCP Server**:
1. Use llama3.2:3b as default for context generation (good speed/quality balance)
2. Set timeout to 300 seconds (5 minutes) to handle large documents
3. Implement retry logic with exponential backoff (3 retries)
4. Process chunks in batches of 10 for efficiency
5. Enable response caching to avoid regenerating same contexts
6. Warm up model during server startup
7. Monitor performance metrics (latency, success rate)
8. Provide fallback to llama3.2:1b if 3b model fails
9. Use async processing for non-blocking operations
10. Log all errors to stderr (MCP requirement)

**Configuration**:
```python
class OllamaConfig(BaseModel):
    """Ollama configuration for OpenRAG."""

    # Model selection
    context_model: str = "llama3.2:3b"
    summary_model: str = "llama3.2:3b"
    fallback_model: str = "llama3.2:1b"

    # Timeouts
    generation_timeout: float = 300.0  # 5 minutes
    warmup_timeout: float = 60.0       # 1 minute

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Performance
    batch_size: int = 10
    enable_caching: bool = True
    cache_size: int = 1000

    # Ollama server
    ollama_host: str = "http://localhost:11434"
```

**Expected Performance**:
- Context generation: 1-3 seconds per chunk (llama3.2:3b)
- Batch of 10 chunks: 10-30 seconds
- 100-chunk document: ~3-5 minutes with batching
- Cache hit: <1ms (instant)
- Model warmup: 5-15 seconds (one-time on startup)

**Common Gotchas**:
- Model must be pulled before use (`ollama pull llama3.2:3b`)
- Ollama server must be running (`ollama serve`)
- First generation after startup is slower (model loading)
- Very long documents may hit context limit (128K tokens for llama3.2)
- GPU recommended for faster generation (but works on CPU)
- Timeout errors common with default settings (increase timeout!)

**Tags**: #ollama #llama3.2 #local-llm #context-generation #summarization #error-handling #timeout #retry-logic #async #caching #performance

---

## 2026-01-16 - Graph RAG Implementation Research

**Context**: Research phase for implementing Graph RAG as a third independent vector store alongside traditional RAG and contextual RAG in the OpenRAG project. Focus on local implementation without cloud dependencies, LangChain integration, architecture patterns, library options, and performance considerations.

**Sources**:
- [Graph RAG & Elasticsearch: Implementing RAG on a Knowledge Graph](https://www.elastic.co/search-labs/blog/rag-graph-traversal)
- [Microsoft GraphRAG GitHub Repository](https://github.com/microsoft/graphrag)
- [What is GraphRAG: Complete guide 2025](https://www.meilisearch.com/blog/graph-rag)
- [Neo4j RAG Tutorial](https://neo4j.com/blog/developer/rag-tutorial/)
- [LightRAG GitHub Repository](https://github.com/HKUDS/LightRAG)
- [Building GraphRAG Locally - Neo4j Blog](https://medium.com/neo4j/building-graphrag-locally-0c8e11752644)
- [Neo4j GraphRAG Python Documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
- [LangChain Blog: Enhancing RAG with Knowledge Graphs](https://www.blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)
- [GraphRAG with LangChain, Gemini and Neo4j](https://medium.com/@vaibhav.agarwal.iitd/building-a-graphrag-system-with-langchain-e63f5e374475)
- [GraphRAG with Neo4j and LangChain - Towards AI](https://pub.towardsai.net/graphrag-explained-building-knowledge-grounded-llm-systems-with-neo4j-and-langchain-017a1820763e)
- [Microsoft GraphRAG Methods](https://microsoft.github.io/graphrag/index/methods/)
- [Structuring Multi-Domain Entity Extraction with Graph RAG + Pydantic](https://www.jellyfishtechnologies.com/structuring-multi-domain-entity-extraction-with-graph-rag-pydantic/)
- [Entity Linking with Relik in LlamaIndex](https://neo4j.com/blog/developer/entity-linking-relationship-extraction-relik-llamaindex/)
- [GraphRAG Complete Guide - DataCamp](https://www.datacamp.com/tutorial/graphrag)
- [RAG vs GraphRAG: Shared Goal & Key Differences](https://memgraph.com/blog/rag-vs-graphrag)
- [Graph RAG vs Traditional RAG - Designveloper](https://www.designveloper.com/blog/graph-rag-vs-traditional-rag/)
- [Traditional RAG vs Graph RAG: Evolution of Retrieval Systems](https://www.analyticsvidhya.com/blog/2025/03/traditional-rag-vs-graph-rag/)
- [HybridRAG: Combining Vector Embeddings with Knowledge Graphs](https://memgraph.com/blog/why-hybridrag)
- [Neo4j GraphRAG Python Package](https://github.com/neo4j/neo4j-graphrag-python)
- [Graph RAG Demystified with NetworkX](https://medium.com/@soumya.chak3/graph-rag-demystified-f73556c65685)
- [NetworkX Challenges: Data Persistency & Large-Scale Analytics](https://memgraph.com/blog/data-persistency-large-scale-data-analytics-and-visualizations-biggest-networkx-challenges)
- [Best Embedding Models for RAG - ZenML](https://www.zenml.io/blog/best-embedding-models-for-rag)
- [5 Best Embedding Models for RAG](https://greennode.ai/blog/best-embedding-models-for-rag)
- [Best Vector Databases for RAG: 2025 Comparison](https://latenode.com/blog/ai-frameworks-technical-infrastructure/vector-databases-embeddings/best-vector-databases-for-rag-complete-2025-comparison-guide)
- [Hybrid Retrieval for GraphRAG - Neo4j Blog](https://neo4j.com/blog/developer/hybrid-retrieval-graphrag-python-package/)
- [LightRAG Official Website](https://lightrag.github.io/)
- [LightRAG: Simple and Fast Alternative to GraphRAG - LearnOpenCV](https://learnopencv.com/lightrag/)
- [TheAiSingularity/graphrag-local-ollama GitHub](https://github.com/TheAiSingularity/graphrag-local-ollama)
- [GraphRAG Local Setup via Ollama - Chi-Sheng Liu](https://chishengliu.com/posts/graphrag-local-ollama/)
- [GraphRAG Local Setup via vLLM and Ollama](https://medium.com/@ysaurabh059/graphrag-local-setup-via-vllm-and-ollama-a-detailed-integration-guide-5d85f18f7fec)
- [LlamaIndex GraphRAG Implementation V2](https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/)
- [RAGDoll: Efficient Offloading-based Online RAG System](https://arxiv.org/html/2504.15302)
- [Hardware Requirements Discussion - Microsoft GraphRAG](https://github.com/microsoft/graphrag/discussions/325)

**Key Findings**:

### 1. Local Implementation Requirements

**Graph RAG is fully implementable locally without cloud services**. Multiple frameworks support local deployment with open-source models:

**Microsoft GraphRAG**:
- Structured, hierarchical approach to RAG with knowledge graph extraction
- Local implementation possible but computationally intensive
- Performance comparison: 36 hours for 19 Markdown files with local models vs. <10 minutes with cloud models
- Requires Python 3.10-3.11 for compatibility

**LightRAG (Recommended for OpenRAG)**:
- Fast, efficient alternative to Microsoft GraphRAG
- Presented at EMNLP 2025 as state-of-the-art framework
- **Performance**: 30% faster than standard RAG (80ms vs 120ms response time)
- **Cost**: 6,000x cheaper than GraphRAG (100 vs 610,000 tokens per query)
- **Architecture**: Dual-level retrieval (local + global + hybrid modes)
- **Storage**: Pluggable backends (NetworkX, Neo4j, PostgreSQL, MongoDB, Milvus)
- **LLM Support**: OpenAI, Ollama, Gemini, Bedrock, HuggingFace
- **Local Deployment**: Full offline operation with Ollama
- **Recommended Requirements**:
  - LLM with at least 32B parameters
  - Context length: 32KB minimum, 64KB recommended
  - Async initialization required: `await rag.initialize_storages()`

**Code-Graph-RAG**:
- Supports both local models (Ollama) and cloud models
- Privacy-focused with no API costs for local models
- Potentially lower accuracy than cloud alternatives

### 2. Local Graph Database Options

**Comparison Table**:

| Database | Type | Pros | Cons | Best For |
|----------|------|------|------|----------|
| **Neo4j** | Native Graph DB | Production-ready, clustering, security, LangChain integration | Heavier resource footprint | Production systems, complex queries |
| **NetworkX** | In-memory Python | Lightweight, simple, fast for small graphs | Memory-intensive (100 bytes/edge), no persistence, not scalable | Development, small datasets (<1M edges) |
| **Memgraph** | Native Graph DB | 5x faster than NetworkX, C++ optimized, NetworkX-compatible API | Less mature ecosystem | High-performance local deployment |
| **PostgreSQL + AGE** | Hybrid | SQL + graph capabilities | Slower than Neo4j in benchmarks | Existing PostgreSQL infrastructure |

**Neo4j Local Options**:
- **Neo4j Desktop**: Recommended for development/non-production
- **Neo4j Embedded**: Not explicitly discussed in latest docs
- **Neo4j Community Edition**: Self-managed local deployment
- Connection: `neo4j://localhost:7687`
- Python package: `neo4j-graphrag` (successor to deprecated `neo4j-genai`)
- Supports vector + graph hybrid search natively

**NetworkX Limitations**:
- **Memory**: ~40GB RAM needed for 30M edges
- **Storage**: In-memory only, requires custom serialization
- **Scale**: Out-of-memory errors beyond 10M nodes
- **Use Case**: Default option for LightRAG file-based storage

**Recommendation for OpenRAG**:
- **Development**: NetworkX (simplest, works with LightRAG out-of-box)
- **Production**: Neo4j Community Edition (scalability, persistence, hybrid search)
- **Performance**: Memgraph (if NetworkX becomes bottleneck)

### 3. LangChain Integration

**LangChain provides comprehensive GraphRAG support**:

**Installation**:
```python
pip install langchain langchain-community langchain-neo4j langchain-experimental
```

**Neo4j Integration Features**:
- **LLMGraphTransformer**: Automated knowledge graph generation from text
- **Neo4jVector**: Dual-purpose vector + graph store
- **Cypher Generation**: Dynamic query generation via LLM
- **Hybrid Search**: Vector similarity + graph traversal combined

**Implementation Pattern**:
```python
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize graph database
graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="password"
)

# Entity extraction and graph construction
transformer = LLMGraphTransformer(llm=your_llm)
graph_documents = transformer.convert_to_graph_documents(documents)

# Vector + Graph hybrid
vectorstore = Neo4jVector(
    embedding=HuggingFaceEmbeddings(),
    graph=graph,
    node_label="Document"
)
```

**MongoDB Alternative**:
- Supports GraphRAG as knowledge graphs instead of vector embeddings
- Enables relationship-aware retrieval and multi-hop reasoning
- Less common than Neo4j in LangChain ecosystem

**LlamaIndex Integration**:
- **GraphRAGExtractor**: Extracts subject-relation-object triples
- **KnowledgeGraphRAGQueryEngine**: Combines graph + vector retrieval
- Async support: `asyncio` and `nest_asyncio` for Jupyter environments
- Parallel processing: `num_workers` parameter for concurrent extraction

### 4. Architecture Patterns: Graph RAG vs Traditional vs Contextual

**Traditional RAG**:
- **Retrieval**: Dense vector search, semantic similarity matching
- **Storage**: Flat vector embeddings in vector database
- **Query**: Single-hop retrieval based on cosine similarity
- **Strength**: Fast, simple, works well for isolated fact retrieval
- **Limitation**: No relationship understanding, isolated chunks, limited multi-document synthesis

**Contextual RAG**:
- **Enhancement**: Adds context to chunks before embedding (prepend document summary, position info)
- **Storage**: Still vector-based, but enriched chunk representations
- **Query**: Semantic search with better context awareness
- **Strength**: Improved chunk relevance, better document-level coherence
- **Limitation**: Still limited to semantic similarity, no explicit relationships

**Graph RAG**:
- **Representation**: Entities (nodes) + Relationships (edges) in knowledge graph
- **Storage**: Graph database with optional vector embeddings for entities
- **Query**: Graph traversal + multi-hop reasoning + semantic search
- **Strength**: Relationship-aware, multi-hop reasoning, explainable paths, evolving knowledge
- **Limitation**: Slower, more complex, higher computational cost

**Key Architectural Differences**:

1. **Entity Extraction**: Graph RAG uses LLM to extract named entities (people, organizations, concepts, events) from text chunks
2. **Relationship Modeling**: Explicit extraction of relationships between entities with descriptions
3. **Graph Construction**: Entities become nodes, relationships become edges with metadata
4. **Community Detection**: Hierarchical clustering of related entities for global queries
5. **Dual Retrieval**:
   - **Local mode**: Find specific entities and immediate neighbors
   - **Global mode**: Retrieve community summaries for broad questions
   - **Hybrid mode**: Combine both approaches (recommended default)

**Indexing Pipeline** (Microsoft GraphRAG):
```
Documents → Chunking (300 tokens) → Entity Extraction → Relationship Extraction
→ Graph Construction → Community Detection → Community Summarization → Storage
```

**Query Mechanisms**:
- **Traditional RAG**: Embed query → Vector similarity → Retrieve top-k chunks → Generate
- **Graph RAG**: Extract query entities → Find similar entities in graph → Expand via edges → Retrieve connected chunks → Combine with vector search → Generate

**Performance Benchmarks**:
- Graph RAG: 86.31% accuracy on RobustQA (3x improvement over traditional RAG)
- Best for: Complex multi-hop queries, relationship-heavy domains
- **Caveat**: Recent research shows GraphRAG sometimes underperforms vanilla RAG on simple tasks

**When to Choose What**:
- **Traditional RAG**: Unstructured text, simple Q&A, speed priority, single-hop facts
- **Contextual RAG**: Need better chunk coherence, document-level understanding, still prioritize speed
- **Graph RAG**: Relationship-rich domains (fraud detection, biomedical, supply chain, social networks), multi-hop reasoning required, explainability important

**Hybrid Approach** (Recommended for OpenRAG):
- Vector search for semantic similarity (fast retrieval)
- Graph traversal for relationship reasoning (accuracy)
- Combine results with reranking
- **Performance**: +35% relevance, +20% latency vs vector-only

### 5. Library Options and Recommendations

**Top Frameworks for Local Graph RAG**:

**1. LightRAG (HIGHEST RECOMMENDATION)**:
```python
# Installation
pip install lightrag-hku

# Features
- Three-tier modular architecture (storage, processing, retrieval)
- Vector storage for embeddings
- Key-value storage for entity summaries
- Graph storage for relationships (NetworkX default)
- Supports OpenAI, Ollama, HuggingFace, Gemini, Bedrock
- Local model deployment: Full offline with Ollama
- Async required: await rag.initialize_storages()

# Storage Backend Options
- NetworkX (default, lightweight)
- Neo4j (production scalability)
- PostgreSQL (hybrid relational+graph)
- MongoDB (document + graph)
- Milvus (vector optimized)
```

**2. Microsoft GraphRAG with Ollama**:
```python
# Repository: TheAiSingularity/graphrag-local-ollama
# Python 3.10-3.11 required

# Setup
conda create -n graphrag python=3.10
conda activate graphrag
pip install graphrag

# Ollama models
ollama pull mistral          # LLM
ollama pull nomic-embed-text # Embeddings

# Configuration (settings.yaml)
llm:
  api_base: http://localhost:11434/v1
  model: mistral

embeddings:
  api_base: http://localhost:11434/api
  model: nomic-embed-text
```

**3. Neo4j GraphRAG Python Package**:
```python
# Installation
pip install neo4j-graphrag  # Replaces deprecated neo4j-genai

# Features
- Official Neo4j package for GraphRAG
- Python 3.9+ required
- Vector search + Cypher generation + graph querying
- Hybrid retrieval built-in
- LangChain compatible

# Connection
from neo4j_graphrag import Neo4jGraphRAG

rag = Neo4jGraphRAG(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="password"
)
```

**4. LlamaIndex GraphRAG**:
```python
# Installation
pip install llama-index llama-index-graph-stores-neo4j

# Features
- GraphRAGExtractor for triple extraction
- KnowledgeGraphRAGQueryEngine for retrieval
- Async support with asyncio/nest_asyncio
- num_workers parameter for parallel processing
- Entity linking with Relik framework
```

**Embedding Models for Graph RAG**:

| Model | License | Dimensions | Context | Speed | Best For |
|-------|---------|-----------|---------|-------|----------|
| **BGE-M3** | MIT (open) | 1024 | 8192 tokens | Medium | Hybrid dense+sparse, production |
| **E5-small** | MIT | 384 | 512 tokens | Very Fast (<30ms) | Low latency, small entities |
| **E5-base-instruct** | MIT | 768 | 512 tokens | Fast (<30ms) | Balanced accuracy/speed |
| **all-MiniLM-L6-v2** | Apache 2.0 | 384 | 256 tokens | Very Fast | Clustering, entity similarity |
| **nomic-embed-text** | Apache 2.0 | 768 | 2048+ tokens | Fast | Ollama integration, local |
| **BAAI/bge-m3** | MIT | 1024 | 8192 tokens | Medium | Multilingual, long docs |

**Installation Example**:
```python
from sentence_transformers import SentenceTransformer

# For entity embeddings
model = SentenceTransformer('BAAI/bge-m3')  # Best for Graph RAG
# or
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight alternative

# Configure vector DB with correct dimensions
# BGE-M3: 1024 dimensions
# MiniLM: 384 dimensions
```

**Vector Databases with Hybrid Search**:

**Weaviate** (Recommended for Hybrid GraphRAG):
- Hybrid search: Dense vector + BM25 keyword matching
- 42% NDCG gain over vector-only, 28% over BM25-only
- Sub-100ms query latency
- Python SDK: `pip install weaviate-client`
- Best balance of features/flexibility

**ChromaDB**:
- Python-first, minimal configuration
- 40M embeddings, 2000+ QPS, <50ms p95 latency (production case)
- Good for prototyping and small-scale

**Milvus**:
- Billions to trillions of vectors
- Cloud-native, distributed architecture
- RESTful API + Python/Java/Go SDKs
- Linear scaling to 10M docs with sharding

**Performance Trade-offs**:
- Fusion (hybrid search): +20% latency, +35% relevance
- Reranking top-100: +15% relevance, 5x cost
- Recall drops 5% beyond 512 dims without quantization

### 6. Performance Considerations

**Resource Requirements**:

**GPU Memory**:
- Large models (Llama-2-70B): 140GB in FP16
- Embedding models + vector DB: ~10GB
- Solution: NVIDIA GH200 (624GB GPU-CPU memory) or offloading to RAM/disk

**CPU/RAM for Graph Processing**:
- NetworkX: ~100 bytes per edge, 40GB RAM for 30M edges
- Neo4j: More memory-efficient with C++ implementation
- Memgraph: 5x faster than NetworkX, optimized storage

**Storage Hierarchy**:
- Hot data: GPU memory (ultra-low latency)
- Warm data: RAM (fast access)
- Cold data: Disk (slower, but unlimited capacity)
- Offloading reduces GPU memory but increases latency

**Latency Benchmarks**:
- LightRAG: 80ms average response time
- Traditional RAG: 120ms average response time
- Cross-encoders (reranking): 100-200ms for 50 candidates
- Vector similarity search: <100ms for most RAG workflows

**Async Processing Strategies**:

**1. Concurrent Entity Extraction** (LlamaIndex):
```python
import asyncio
import nest_asyncio  # For Jupyter

# Enable nested event loops
nest_asyncio.apply()

# Parallel extraction
extractor = GraphRAGExtractor(
    llm=your_llm,
    num_workers=4  # Parallel processing
)

# Async initialization (LightRAG)
await rag.initialize_storages()
```

**2. Batch Processing**:
- Microsoft GraphRAG: Process chunks in parallel during indexing
- LightRAG: Concurrent pipelines for text and multimodal processing
- Recommended batch sizes: 10-50 chunks depending on memory

**3. Async Query Processing** (Morphik pattern):
```python
async def query_with_graph(query: str):
    # Extract entities from query
    entities = await extract_entities(query)

    # Find similar entities (concurrent)
    similar = await find_similar_entities_in_graph(entities)

    # Expand to related entities (graph traversal)
    expanded = await expand_related_entities(similar)

    # Retrieve chunks (concurrent)
    chunks = await retrieve_chunks(expanded)

    # Combine with vector search
    vector_results = await vector_search(query)

    return combine_results(chunks, vector_results)
```

**Scalability Patterns**:

**Small Scale** (<10K documents, <100K entities):
- NetworkX in-memory graphs
- ChromaDB for vectors
- Single machine deployment
- Expected latency: <100ms

**Medium Scale** (10K-1M documents, 100K-10M entities):
- Neo4j Community Edition
- Weaviate or Milvus for vectors
- Hybrid search mandatory
- Expected latency: 100-300ms
- Sharding recommended

**Large Scale** (>1M documents, >10M entities):
- Neo4j Enterprise or Memgraph
- Milvus distributed cluster
- GPU acceleration for embedding generation
- Quantization for memory efficiency (5% recall drop acceptable)
- Expected latency: 200-500ms

**Common Performance Bottlenecks**:
1. **Entity extraction**: Most time-consuming step (hours for 200-300 page docs)
2. **Graph construction**: Memory-intensive for NetworkX
3. **Community detection**: Scales poorly beyond 1M nodes
4. **Cross-encoder reranking**: 5x cost vs. retrieval

**Optimization Strategies**:
- Cache extracted entities (avoid re-processing)
- Use smaller, task-specific models (Relik framework for entity linking)
- Incremental graph updates (avoid full rebuilds)
- Async/parallel processing wherever possible
- Hybrid search only when relationships matter (fall back to vector-only for simple queries)

### 7. Implementation Recommendations for OpenRAG

**Phase 1: MVP (Recommended)**:
- **Framework**: LightRAG (fastest, simplest, SOTA)
- **Graph Storage**: NetworkX (built-in, no setup)
- **Vector Storage**: ChromaDB (existing, Python-first)
- **Embeddings**: all-MiniLM-L6-v2 (lightweight) or BGE-M3 (better quality)
- **LLM**: Ollama with Mistral or Llama 3.2 (already integrated)
- **Query Mode**: Hybrid (local + global + vector)

**Example Implementation**:
```python
from lightrag import LightRAG
from sentence_transformers import SentenceTransformer

# Initialize with Ollama
rag = LightRAG(
    llm_model="ollama/mistral",
    embedding_model="BAAI/bge-m3",
    graph_storage="networkx",  # Default
    vector_storage="chromadb",
    ollama_host="http://localhost:11434"
)

# Async initialization
await rag.initialize_storages()

# Indexing
await rag.insert(documents)

# Querying (hybrid mode recommended)
results = await rag.query(
    query="your question",
    mode="hybrid"  # Combines local, global, and vector search
)
```

**Phase 2: Production (If MVP Successful)**:
- **Framework**: Keep LightRAG or migrate to Neo4j GraphRAG package
- **Graph Storage**: Neo4j Community Edition (persistence, scalability)
- **Vector Storage**: Weaviate (hybrid search, <100ms latency)
- **Embeddings**: BGE-M3 (dense+sparse hybrid)
- **LLM**: Keep Ollama (privacy) or add cloud option (performance)
- **Async**: Full async pipeline with concurrent extraction

**Integration with Existing OpenRAG Architecture**:
```python
# Existing: TraditionalRAG, ContextualRAG
# Add: GraphRAG

class GraphRAG:
    """Graph-based RAG using knowledge graphs for relationship-aware retrieval."""

    def __init__(self, config: GraphRAGConfig):
        self.lightrag = LightRAG(
            llm_model=config.llm_model,
            embedding_model=config.embedding_model,
            graph_storage=config.graph_storage,
            vector_storage=config.vector_storage
        )

    async def initialize(self):
        await self.lightrag.initialize_storages()

    async def insert_documents(self, documents: list[Document]):
        """Extract entities, build graph, and index."""
        await self.lightrag.insert(documents)

    async def query(self, query: str, mode: str = "hybrid") -> QueryResult:
        """
        Query modes:
        - local: Find specific entities and immediate neighbors
        - global: Retrieve community summaries for broad questions
        - hybrid: Combine both (recommended)
        """
        return await self.lightrag.query(query, mode=mode)
```

**Configuration**:
```python
class GraphRAGConfig(BaseModel):
    """Configuration for Graph RAG."""

    # Framework selection
    framework: str = "lightrag"  # or "microsoft_graphrag", "neo4j_graphrag"

    # Storage backends
    graph_storage: str = "networkx"  # or "neo4j", "memgraph"
    vector_storage: str = "chromadb"  # or "weaviate", "milvus"

    # Models
    llm_model: str = "ollama/mistral"
    embedding_model: str = "BAAI/bge-m3"
    ollama_host: str = "http://localhost:11434"

    # Query settings
    default_mode: str = "hybrid"  # local, global, hybrid
    enable_reranking: bool = False  # 5x cost, +15% relevance

    # Performance
    num_workers: int = 4  # Parallel entity extraction
    max_entities_per_chunk: int = 20
    community_detection_enabled: bool = True

    # Resource limits
    max_graph_nodes: int = 1_000_000
    max_edges: int = 10_000_000
```

**Testing Strategy**:
1. Unit tests for entity extraction accuracy
2. Integration tests for graph construction
3. Benchmark query latency vs traditional/contextual RAG
4. Evaluate on multi-hop question datasets (RobustQA, HotpotQA)
5. Memory profiling for large document collections

**Migration Path**:
```
Traditional RAG (existing) → Add Contextual RAG (done) → Add Graph RAG (new)
                                                              ↓
                                                   NetworkX (MVP)
                                                              ↓
                                           Neo4j (if production scaling needed)
```

### 8. Key Technical Considerations

**Entity Extraction Quality**:
- LLM quality matters: Minimum 32B parameters recommended for Graph RAG
- Prompt engineering crucial for consistent entity/relationship extraction
- Pydantic schemas improve extraction structure
- Trade-off: Accuracy vs. processing time (hours for large docs)

**Graph Database Selection Matrix**:
```
Development/MVP → NetworkX (fastest to start)
    ↓ If: Memory issues OR need persistence
Neo4j Desktop
    ↓ If: Production deployment needed
Neo4j Community Edition
    ↓ If: Performance bottleneck
Memgraph (5x faster, NetworkX-compatible API)
```

**Hybrid Search Architecture**:
```
User Query
    ├─→ Vector Search (semantic similarity)
    │   └─→ Top-k chunks
    │
    ├─→ Graph Search (entity extraction + traversal)
    │   ├─→ Local: Specific entities + neighbors
    │   └─→ Global: Community summaries
    │
    └─→ Fusion (combine results)
        └─→ Optional: Reranking (cross-encoder)
            └─→ LLM Generation
```

**Cost-Benefit Analysis**:
- **Graph RAG Wins**: Multi-hop queries, relationship-heavy domains, explainability needed
- **Traditional RAG Wins**: Simple facts, speed priority, cost-sensitive
- **Contextual RAG Wins**: Better chunk coherence without graph complexity
- **Hybrid Wins**: Best accuracy, acceptable latency increase (+20%), production systems

**Common Pitfalls**:
1. Using NetworkX for >10M edges (memory explosion)
2. Synchronous entity extraction (hours of processing)
3. Over-engineering: Graph RAG not always better than traditional RAG
4. Ignoring embedding model dimensions (mismatch with vector DB)
5. No caching strategy (re-extracting entities on every run)

**Implementation Notes**:
- Start with LightRAG + NetworkX for rapid prototyping
- Use Ollama for local LLM (privacy, no API costs)
- BGE-M3 embeddings provide best quality for Graph RAG
- Hybrid query mode recommended as default
- Monitor memory usage closely if using NetworkX
- Plan migration to Neo4j if graph exceeds 1M nodes
- Async/await essential for acceptable performance
- Consider Graph RAG as complementary to traditional RAG, not replacement

**Tags**: #graph-rag #knowledge-graph #lightrag #microsoft-graphrag #neo4j #networkx #langchain #llamaindex #entity-extraction #relationship-modeling #hybrid-search #ollama #local-deployment #embedding-models #bge-m3 #async #performance #scalability #weaviate #chromadb #multi-hop-reasoning #dual-retrieval

---

*Last Updated: 2026-01-16*
*Researcher: Claude (Research Agent)*
*Status: Graph RAG implementation research complete - Ready for MVP development*

---

## 2026-01-16 - Graph RAG Implementation Complete

**Context**: Completed implementation of Graph RAG as the third independent RAG strategy in OpenRAG, running in parallel with Traditional and Contextual RAG.

**Implementation Summary**:

### Architecture Decisions

**Production-Grade Stack** (vs MVP):
- **Neo4j** instead of NetworkX for graph database
  - Rationale: User explicitly requested production-ready solution
  - Benefits: ACID transactions, distributed queries, enterprise scalability
  - Trade-offs: Additional dependency vs. in-memory NetworkX

**Parallel RAG Collections**:
- Three independent ChromaDB collections: `documents_traditional_v1`, `documents_contextual_v1`, `documents_graph_v1`
- Each collection stores different embeddings:
  - Traditional: Raw chunk content
  - Contextual: Chunk + document context
  - Graph: Original content (graph structure in Neo4j)

**Hybrid Search Strategy**:
```
Query → Graph RAG
    ├─→ 1. Vector Search (ChromaDB)
    │      └─→ Initial top-k chunks
    │
    ├─→ 2. Entity Extraction (optional, from query)
    │
    ├─→ 3. Graph Traversal (Neo4j Cypher)
    │      └─→ Find related chunks via entity relationships
    │      └─→ Max hops: configurable (1-5, default: 2)
    │
    └─→ 4. Result Fusion & Re-ranking
           └─→ Return combined results
```

### Implementation Details

**Files Created**:
1. `src/openrag/core/graph_processor.py` (527 lines)
   - GraphProcessor class for entity extraction
   - Ollama-based LLM extraction with XML structured prompts
   - Fallback to regex patterns when Ollama unavailable
   - Neo4j graph storage with async operations
   - Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, DATE, EVENT
   - Relationship types: WORKS_AT, LOCATED_IN, PARTICIPATES_IN, etc.

2. `src/openrag/core/graph_vector_store.py` (455 lines)
   - Extends ContextualVectorStore
   - Manages three RAG collections
   - Implements hybrid graph search (_graph_search method)
   - Cypher query for multi-hop traversal
   - Graph expansion via Neo4j relationships

3. `src/openrag/models/graph_schemas.py` (241 lines)
   - Entity: Pydantic model for extracted entities
   - Relationship: Connections between entities
   - GraphChunk: Enhanced DocumentChunk with graph data
   - GraphDocument: Complete document with graph metadata
   - GraphQueryResult: Query results with graph context

**Files Modified**:
1. `src/openrag/config.py`
   - Added Neo4j configuration (URI, credentials, database)
   - Graph RAG settings (entity_model, max_hops, batch_size)
   - Properties for all new settings

2. `src/openrag/models/contextual_schemas.py`
   - Extended RAGType enum: TRADITIONAL, CONTEXTUAL, GRAPH

3. `src/openrag/tools/ingest.py`
   - Added graph_processor parameter
   - Background graph processing with _process_graph_async
   - Parallel processing: Traditional (sync), Contextual (async), Graph (async)

4. `src/openrag/tools/query.py`
   - Added max_hops parameter for graph queries
   - Support for rag_type="graph"
   - Validation for all three RAG types

5. `src/openrag/server.py`
   - GraphProcessor initialization with error handling
   - Updated tool schemas to include "graph" option
   - Added max_hops parameter to query_documents tool
   - Cleanup logic for Neo4j connections

6. `requirements.txt`
   - Added: neo4j>=5.0.0, neo4j-graphrag

**Tests Created**:
1. `tests/test_graph_processor.py` (330 lines)
   - Unit tests for entity extraction
   - Relationship parsing tests
   - Neo4j storage mocking
   - Fallback extraction tests

2. `tests/test_graph_vector_store.py` (340 lines)
   - GraphVectorStore initialization
   - Adding graph documents
   - Graph search with expansion
   - Multi-collection management

3. `tests/test_graph_integration.py` (380 lines)
   - End-to-end workflow tests
   - Parallel RAG types validation
   - Graph expansion integration tests

### Technical Achievements

**Entity Extraction**:
- LLM-based extraction using Ollama
- Structured XML prompts for consistent parsing
- Confidence scoring (0.9 for LLM, 0.5 for regex fallback)
- Batch processing support

**Graph Storage**:
- Async Neo4j operations throughout
- Entity uniqueness constraints
- Chunk ID indexing
- MERGE operations prevent duplicates
- Relationship creation with confidence scores

**Hybrid Search**:
- Combines vector similarity (ChromaDB) with graph traversal (Neo4j)
- Configurable max_hops for traversal depth
- Re-ranking: initial results keep vector scores, expanded chunks get lower scores
- Limit: 20 expanded chunks to control latency

**Background Processing**:
- Graph extraction non-blocking (uses BackgroundTaskManager)
- Traditional RAG available immediately
- Contextual and Graph process asynchronously
- Status tracking in tool responses

### Key Design Patterns

**Graceful Degradation**:
- Graph RAG disabled if Neo4j unavailable → falls back to vector search
- Ollama failure → regex fallback for entity extraction
- Each RAG type independent: failures don't cascade

**Extension Pattern**:
- GraphVectorStore extends ContextualVectorStore
- ContextualVectorStore extends VectorStore
- Clean inheritance hierarchy
- Minimal code duplication

**Configuration**:
- All settings in config.py with validation
- Environment variable support (.env)
- Sensible defaults for all parameters
- Property accessors for backward compatibility

### Performance Considerations

**Trade-offs**:
- Graph RAG slower than traditional (entity extraction + Neo4j queries)
- Compensated by background processing
- User queries fast (vector search + graph traversal)
- Memory: ChromaDB (3 collections) + Neo4j database

**Optimizations**:
- Async/await throughout
- Batch processing for entities
- Neo4j indexes on chunk_id
- Limited graph expansion (max 20 chunks)
- Re-ranking avoids redundant embedding generation

### Validation

**Ruff Checks**: ✅ All passed
- Line length compliance (100 chars)
- Import ordering
- Type annotations
- Docstring coverage

**Manual Testing**:
- Cannot run pytest without Python 3.10+ (current: 3.9)
- Tests are comprehensive and follow pytest best practices
- Will validate in final testing step

### Documentation Updated

**README.md**:
- Updated title: "Multi-Strategy RAG MCP Server"
- Added Graph RAG feature description
- Configuration examples with Neo4j
- Architecture diagram with all components
- RAG strategy comparison table

**Lab Journal**:
- This entry documenting complete implementation
- Technical decisions and rationale
- Architecture patterns used

### Lessons Learned

1. **Production vs MVP**: User's explicit request for "production" system (Neo4j) significantly increased scope but provides better scalability

2. **Async Complexity**: Async/await essential for acceptable performance but increases testing complexity

3. **Graceful Degradation**: Making each RAG type optional and independent improves robustness

4. **Test Coverage**: Created 1,050+ lines of tests covering unit, integration, and edge cases

5. **Configuration Management**: Centralized config.py with pydantic validation prevents runtime errors

### Next Steps

**Task 8**: Create feature branch and pull request
- Branch name: `feature/graph-rag-implementation`
- Comprehensive PR description with implementation details

**Task 9**: Final validation
- Run pytest suite (requires Python 3.10+ environment)
- Manual testing with actual Neo4j instance
- Integration testing with Ollama

**Tags**: #graph-rag #neo4j #production-implementation #entity-extraction #hybrid-search #async #multi-strategy-rag #chromadb #ollama #knowledge-graph #completed-implementation

---

*Implementation Date: 2026-01-16*
*Developer: Claude Code*
*Status: Implementation Complete - Ready for PR and Testing*
