# Tools & Utilities

This folder contains debugging, diagnostic, and utility tools for the RAG system development and troubleshooting.

## Categories:

### CHECKING: **Diagnostics** (Database & System Analysis):
- `check_enhanced_db.py` - Enhanced database inspection and validation
- `check_db.py` - Basic database connectivity and content check  
- `check_all_tables.py` - List all tables in database
- `check_table_structure.py` - Analyze table structure and create test tables
- `check_embedding_dim.py` - Verify embedding dimensions
- `compare_table_structures.py` - Compare original vs working table structures
- `debug_table_structure.py` - Detailed table structure debugging

### TOOLS: **Debug Tools** (Testing & Troubleshooting):
- `test_retrieval_debug.py` - Basic vector similarity search testing
- `test_working_table.py` - Testing with working LlamaIndex table
- `test_exact_config.py` - Testing with exact ingestion configuration
- `test_minimal_config.py` - Minimal LlamaIndex configuration testing
- `test_clean_table.py` - Testing with clean table structures
- `test_fresh_ingestion.py` - Testing fresh LlamaIndex data ingestion
- `test_direct_vector_search.py` - Direct PostgreSQL vector similarity testing
- `comprehensive_debug.py` - Comprehensive debugging of all components

### INFO: **Analysis Tools**:
- `analyze_data.py` - Simple database content analysis
- `custom_retriever.py` - Custom PostgreSQL vector retriever (working solution)

### FOLDER: **Reference Files**:
- `tools_broken.py` - Backup of tools.py with LlamaIndex PGVectorStore (non-functional)
- `tools_fixed.py` - Fixed version with custom retriever

## Key Findings:

1. **LlamaIndex PGVectorStore Issue**: The standard LlamaIndex PGVectorStore had compatibility issues with our PostgreSQL setup, returning 0 results despite having valid data.

2. **Custom Retriever Solution**: Direct PostgreSQL vector similarity queries work perfectly, leading to the CustomPGVectorRetriever implementation.

3. **Vector Storage Format**: Embeddings are properly stored as PostgreSQL vector(768) type with correct similarity calculations.

4. **Table Naming**: LlamaIndex automatically prefixes table names with "data_".

## Usage:

These files are kept for reference and can be used to:
- Test vector similarity search functionality
- Debug database connection issues
- Validate embedding storage and retrieval
- Compare different retrieval approaches

Run any of the test files with:
```bash
pipenv run python debug/<filename>
```
