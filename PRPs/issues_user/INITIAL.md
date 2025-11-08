## FEATURE:

My overall goal is to create an open-source MCP server that builds a Retrieval Augmented Generation (RAG) system over personal documents. The system should enable users to efficiently search, retrieve, and generate information based on their own data using advanced language models. The database will be saved locally on the user's machine to ensure privacy and security. The project will later cover different RAG types. For the first implementation we will focus on traditional RAG.

The following features must be available in the initial version:

- A document will be handed over from the LLM to which this future MCP servers is connected to in the form of a .txt file. 
- The document must be chunked into smaller pieces and embedded using an embedding model.
- The embedded chunks must be stored in a vector database locally on the user's machine.
- The user can specify the exact location where the vector database is stored.
- Several metadata fields must be stored alongside the embedded chunks, including at least_
    - chunk ID
    - source document name
    - chunk text
    - date and time when the chunk was created
- The user can query the vector database with a natural language query.
- The system retrieves the most relevant chunks based on the user's query.
- The system returns the retrieved chunks to the LLM for further processing.

## EXAMPLES:

In the folder ./examples the following files are available from another project that can be used as inspiration:

- document_manager.py: This class handles the storage, removing and updating of the metadata for any documents that are added to the vector database.
- document_processor.py: This class is responsible for chunking the documents into smaller pieces and embedding them using an embedding model.
- document_watcher.py: This class monitors a specified directory for new documents and triggers the processing and storage of those documents in the vector database.
- rag_engine.py: Initializes the RAG system, i.e. initializes document_processor, document_manager. But it also ingests new documents, handles user queries, retrieves relevant chunks from the vector database, and returns them to the LLM for further processing.
- vector_store.py: This class manages the vector database, including adding, retrieving (similarity search), and deleting embedded chunks and their associated metadata in ChromaDB.


## DOCUMENTATION:

- ChromaDB documentation: https://docs.trychroma.com/docs/overview/introduction
- Ollama documentation: https://docs.ollama.com/
- 

## OTHER CONSIDERATIONS:

- CHROMADB: The vector database should be implemented using ChromaDB, an open-source vector database that is easy to use and integrates well with Python.
- EMBEDDING MODEL: There should be the option for a fast and simple embedding model but also a more powerful one
- MCP SERVER DEPLOYMENT: Think about what is the best way to deploy the MCP server later. Add your results to a lab_journal.md in the ./docs folder.
- TESTING: I need the option to easily test the different components of the RAG system. Write unit tests for the main classes and functions in the ./tests folder. 