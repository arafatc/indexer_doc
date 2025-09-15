# RAG Data Ingestion System

A sophisticated document ingestion system for Retrieval-Augmented Generation (RAG) applications, featuring multiple chunking strategies and advanced document processing capabilities.

## 🚀 Features

- **5 Advanced Chunking Strategies** - Choose the optimal strategy for your use case
- **Multi-format Support** - Process PDF, DOCX, and other document formats
- **Table & Image Extraction** - Extract structured data from complex documents
- **LLM Integration** - Support for Ollama and OpenAI models
- **PostgreSQL Vector Storage** - Efficient embedding storage with pgvector
- **Phoenix Observability** - Built-in tracing and monitoring
- **Production Ready** - Optimized for performance and scalability

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Ollama (for local LLM) or OpenAI API key
- Pipenv for dependency management

## 🛠 Installation

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd indexer_doc
pipenv install
pipenv shell
```

### 2. Database Setup

```bash
# Install PostgreSQL and pgvector extension
# Create database
createdb rag_db

# Enable pgvector extension
psql rag_db -c "CREATE EXTENSION vector;"
```

### 3. Ollama Setup (Optional)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

### 4. Configuration

Copy and customize the environment file:

```bash
cp .env.example .env
```

## ⚙️ Configuration

The system is configured via the `.env` file. Key settings include:

### Chunking Strategies

Choose from 5 sophisticated strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `basic` | Fast & reliable fixed-size chunks | Quick setup, testing |
| `structure_aware` | Preserves document structure | Structured documents |
| `semantic` | Embedding-based coherence | Balanced performance |
| `hierarchical` | Multi-level relationships | Complex documents |
| `contextual_rag` | AI-powered metadata extraction | Maximum quality |

### Performance Profiles

#### Development/Testing (Fast Setup)
```env
CHUNKING_STRATEGY=basic
ENABLE_TABLE_EXTRACTION=false
LLM_PROVIDER=ollama
BATCH_SIZE=5
```

#### Production (Recommended)
```env
CHUNKING_STRATEGY=semantic
ENABLE_TABLE_EXTRACTION=true
LLM_PROVIDER=ollama
BATCH_SIZE=3
REQUEST_DELAY=0.1
```

#### Maximum Quality
```env
CHUNKING_STRATEGY=contextual_rag
ENABLE_CONTEXTUAL_RAG=true
ENABLE_TABLE_EXTRACTION=true
LLM_PROVIDER=openai
```

## 🚀 Usage

### Basic Commands

```bash
# Basic RAG ingestion
pipenv run python src/ingestion/ingest_enhanced_cli.py --chunking-strategy basic --enable-tables

# Semantic chunking with table extraction
pipenv run python src/ingestion/ingest_enhanced_cli.py --chunking-strategy semantic --enable-tables

# Hierarchical chunking
pipenv run python src/ingestion/ingest_enhanced_cli.py --chunking-strategy hierarchical --enable-tables

# Structure-aware processing
pipenv run python src/ingestion/ingest_enhanced_cli.py --chunking-strategy structure_aware --enable-tables

# Contextual RAG (AI-powered)
pipenv run python src/ingestion/ingest_enhanced_cli.py --chunking-strategy contextual_rag --enable-tables
```

### Advanced Options

```bash
# Custom chunk size and overlap
pipenv run python src/ingestion/ingest_enhanced_cli.py \
  --chunking-strategy semantic \
  --chunk-size 2048 \
  --chunk-overlap 100 \
  --enable-tables

# Enable image extraction
pipenv run python src/ingestion/ingest_enhanced_cli.py \
  --chunking-strategy semantic \
  --enable-images \
  --enable-tables

# Batch processing with custom delays
pipenv run python src/ingestion/ingest_enhanced_cli.py \
  --chunking-strategy hierarchical \
  --batch-size 5 \
  --enable-tables
```

## 📁 Project Structure

```
indexer_doc/
├── .env                     # Configuration file
├── README.md               # This file
├── Pipfile                 # Python dependencies
├── src/
│   └── ingestion/
│       └── ingest_enhanced_cli.py    # Main ingestion script
├── data/
│   └── raw/                # Input documents
├── tools/
│   └── README.md          # Tools documentation
└── docs/                  # Additional documentation
```

## 🔧 Environment Variables Reference

### Core Settings
```env
# Chunking Strategy
CHUNKING_STRATEGY=semantic
CHUNK_SIZE=1024
CHUNK_OVERLAP=50

# Features
ENABLE_TABLE_EXTRACTION=true
ENABLE_IMAGE_EXTRACTION=false
ENABLE_CONTEXTUAL_RAG=false

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:1b
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_DIM=768
```

### Database Configuration
```env
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=postgres
PG_DATABASE=rag_db
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db
```

### Performance Tuning
```env
BATCH_SIZE=3
REQUEST_DELAY=0.1
CHUNK_BATCH_SIZE=5
FILE_BATCH_DELAY=1.0
CHUNK_BATCH_DELAY=0.5
```

## 📊 Monitoring with Phoenix

The system includes Arize Phoenix integration for observability:

```env
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

Access Phoenix UI at `http://localhost:6006` to monitor:
- Request traces
- Embedding performance
- Token usage
- Error rates

## 🏗 Deployment

### Development
```bash
# Start all services
docker-compose up -d postgres
ollama serve &
phoenix serve &

# Run ingestion
pipenv run python src/ingestion/ingest_enhanced_cli.py --chunking-strategy semantic --enable-tables
```

### Production

1. **Database Setup**
```bash
# Use managed PostgreSQL service
# Ensure pgvector extension is enabled
```

2. **Model Deployment**
```bash
# For Ollama
ollama serve --host 0.0.0.0 --port 11434

# Or configure OpenAI
export OPENAI_API_KEY="your-key"
```

3. **Environment Configuration**
```bash
# Use production values
CHUNKING_STRATEGY=semantic
LLM_PROVIDER=ollama
BATCH_SIZE=3
STORE_EMBEDDINGS=true
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
```bash
# Verify PostgreSQL is running
pg_isready -h localhost -p 5432

# Check pgvector extension
psql rag_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

2. **Ollama Model Issues**
```bash
# Check available models
ollama list

# Pull missing models
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

3. **Memory Issues**
```bash
# Reduce batch size
BATCH_SIZE=1
CHUNK_BATCH_SIZE=3
```

### Performance Optimization

- **Large Documents**: Increase `CHUNK_SIZE=2048`, `CHUNK_OVERLAP=100`
- **Fast Processing**: Increase `BATCH_SIZE=5`, `CHUNK_BATCH_SIZE=10`
- **Resource Constrained**: Use `basic` chunking strategy

## 📚 Documentation

- [Tools Documentation](./tools/README.md)
- [Configuration Guide](./docs/configuration.md)
- [API Reference](./docs/api.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Arize Phoenix](https://docs.arize.com/phoenix/)
- [Ollama Models](https://ollama.ai/library)

---

*For detailed tool documentation, see [tools/README.md](./tools/README.md)*