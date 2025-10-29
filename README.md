# OpenWebUI Agentic RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with multi-agent orchestration, conversation memory, observability, and comprehensive evaluation capabilities.

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12+-6B46C1)](https://www.llamaindex.ai/)

## 🌟 Features


- **🤖 Advanced RAG Pipeline** - Contextual retrieval with Anthropic-style enhancement
- **🔄 Re-ranking** - Cohere rerank-english-v3.0 with similarity-based fallback
- **👥 Multi-Agent System** - CrewAI orchestration with specialized agents
- **💬 Conversation Memory** - Session-based chat with context-aware responses
- **📊 Observability** - Arize Phoenix for LLM tracing and performance monitoring
- **📈 Evaluation** - RAGAs metrics (faithfulness, relevancy, precision, recall)
- **🎨 Modern UI** - OpenWebUI chat interface
- **📄 Document Processing** - Docling for advanced PDF parsing
- **🗄️ Vector Storage** - PostgreSQL with PGVector extension

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/RakshithaMallabadi/AgenticRAG-with-OpenWebUI.git
cd AgenticRAG-with-OpenWebUI
```

### 2. Start All Services

```bash
docker compose up -d
```

### 3. Wait for Services to Initialize

```bash
# Check service health
docker compose ps

# View FastAPI logs
docker compose logs -f fastapi
```

### 4. Ingest Documents

```bash
# Place your documents in the data/ folder
cp your_documents.pdf data/

# Trigger ingestion
curl -X POST "http://localhost:4000/ingest"
```

### 5. Access the Applications

- **OpenWebUI (Chat Interface)**: http://localhost:3000
- **FastAPI Backend**: http://localhost:4000
- **API Documentation**: http://localhost:4000/docs
- **Phoenix Observability**: http://localhost:6006

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          OpenWebUI (Port 3000)                       │
│                         User Chat Interface                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ OpenAI-compatible API
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (Port 4000)                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Contextual RAG   │  │ Conversation     │  │ CrewAI Agents   │  │
│  │ + Re-ranking     │  │ Memory           │  │ Orchestration   │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ LlamaIndex       │  │ Arize Phoenix    │  │ RAGAs           │  │
│  │ Query Engine     │  │ Observability    │  │ Evaluation      │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
└───┬──────────────────────┬──────────────────────┬──────────────────┘
    │                      │                      │
    ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ PostgreSQL  │    │ Ollama       │    │ Document Store │
│ + PGVector  │    │ llama3.2:1b  │    │ (data/)        │
│ (Port 5432) │    │ (Port 11434) │    └────────────────┘
└─────────────┘    └──────────────┘
```

For detailed architecture information, see [Architecture.md](Architecture.md).

## 📦 Prerequisites

### Required Software

- **Docker Desktop** (v20.10+) - [Download](https://www.docker.com/products/docker-desktop)
- **Docker Compose** (v2.0+) - Included with Docker Desktop
- **Git** - For cloning the repository

### System Requirements

- **RAM**: 4-5 GB minimum
- **CPU**: 2+ cores recommended
- **Disk**: ~5 GB for models and database
- **OS**: macOS, Linux, or Windows with WSL2

### Optional

- **Cohere API Key** - For advanced re-ranking ([Get one here](https://cohere.com/))

## 💻 Installation

### Method 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/RakshithaMallabadi/AgenticRAG-with-OpenWebUI.git
cd AgenticRAG-with-OpenWebUI

# 2. Configure environment (optional)
cp .env.example .env
# Edit .env with your configurations

# 3. Start all services
docker compose up -d

# 4. Check service health
docker compose ps
```

### Method 2: Local Development

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL and Ollama separately
docker compose up postgres ollama -d

# 4. Run FastAPI
uvicorn app:app --host 0.0.0.0 --port 4000 --reload
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:1b

# Database
DATABASE_URL=postgresql://raguser:ragpass@postgres:5432/ragdb

# Phoenix Observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=openwebui-rag

# Optional: Cohere Re-ranking (for better results)
COHERE_API_KEY=your-cohere-api-key-here
```

### Model Configuration

By default, the system uses `llama3.2:1b` (1.3 GB) for lower memory usage. To use larger models:

```yaml
# In docker-compose.yml, update OLLAMA_MODEL
environment:
  - OLLAMA_MODEL=llama3.2  # 2.0 GB, better quality
```

### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). To change:

```python
# In app.py, update embed_model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # Alternative model
    device="cpu"
)
```

## 📖 Usage

### Document Ingestion

#### Via API

```bash
# Ingest all documents in data/ folder
curl -X POST "http://localhost:4000/ingest"
```

#### Supported Formats

- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text Files (`.txt`)
- Markdown (`.md`)

### Querying the System

#### Via OpenWebUI (Recommended)

1. Open http://localhost:3000
2. Start chatting - all features are automatically enabled
3. Ask questions about your documents

#### Via API

```bash
# Simple query
curl -X POST "http://localhost:4000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main procurement requirements?",
    "top_k": 5
  }'

# Conversational query with memory
curl -X POST "http://localhost:4000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tell me about the document",
    "session_id": "user123",
    "top_k": 5
  }'
```

### Evaluation

```bash
# Single query evaluation
curl -X POST "http://localhost:4000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is procurement?",
    "ground_truth": "Procurement is the process of acquiring goods and services",
    "top_k": 5
  }'

# Batch evaluation
curl -X POST "http://localhost:4000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is procurement?",
      "Who approves contracts?"
    ]
  }'
```

## 🔌 API Endpoints

### Core RAG Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Stateless RAG with re-ranking |
| `/chat` | POST | Conversational RAG with memory |
| `/ingest` | POST | Document ingestion |
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |

### Memory Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/history/{session_id}` | GET | View conversation history |
| `/chat/clear` | POST | Clear session memory |
| `/chat/sessions` | GET | List active sessions |
| `/chat/sessions` | DELETE | Clear all sessions |

### Observability

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/phoenix/status` | GET | Phoenix observability status |
| `/phoenix/traces` | GET | View LLM traces |
| `/phoenix/metrics` | GET | Performance metrics |

### Evaluation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ragas/status` | GET | RAGAs evaluation status |
| `/evaluate` | POST | Single query evaluation |
| `/evaluate/batch` | POST | Batch evaluation |

For complete API documentation, visit http://localhost:4000/docs after starting the services.

## 📚 Documentation

### Primary Documents

- **[Architecture.md](Architecture.md)** - Comprehensive system architecture and design
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Complete project overview and status
- **[CREWAI_IMPLEMENTATION.md](CREWAI_IMPLEMENTATION.md)** - Multi-agent system details
- **[PHOENIX_OBSERVABILITY.md](PHOENIX_OBSERVABILITY.md)** - Observability setup and usage
- **[RAGAS_EVALUATION.md](RAGAS_EVALUATION.md)** - Evaluation framework guide
- **[OPENWEBUI_CREWAI_INTEGRATION.md](OPENWEBUI_CREWAI_INTEGRATION.md)** - UI integration details

### Key Technologies

- **LlamaIndex** - RAG orchestration framework
- **CrewAI** - Multi-agent orchestration
- **Arize Phoenix** - LLM observability and tracing
- **RAGAs** - RAG evaluation metrics
- **Ollama** - Local LLM hosting
- **OpenWebUI** - Chat interface
- **PGVector** - Vector database
- **Docling** - Document processing

## 🔧 Troubleshooting

### Services Won't Start

```bash
# Check logs
docker compose logs

# Rebuild containers
docker compose build --no-cache
docker compose up -d
```

### FastAPI Unhealthy

```bash
# Check FastAPI logs
docker compose logs fastapi | tail -50

# Common issues:
# - Ollama not ready: wait 2 minutes
# - PostgreSQL not ready: check postgres logs
# - Missing dependencies: rebuild image
```

### No Results from RAG

```bash
# Check if documents are ingested
curl http://localhost:4000/health | jq

# Check database
docker compose exec postgres psql -U raguser -d ragdb \
  -c "SELECT COUNT(*) FROM data_llamaindex_documents;"

# Re-ingest if needed
curl -X POST http://localhost:4000/ingest
```

### Remove Duplicate Documents

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U raguser -d ragdb

# Remove duplicates
DELETE FROM data_llamaindex_documents a USING (
    SELECT MIN(id) as id, text, metadata_->>'source' as source
    FROM data_llamaindex_documents
    GROUP BY text, metadata_->>'source'
    HAVING COUNT(*) > 1
) b
WHERE a.text = b.text 
AND a.metadata_->>'source' = b.source 
AND a.id <> b.id;
```

### OpenWebUI Not Connecting

```bash
# Check environment
docker compose exec openwebui env | grep OPENAI

# Restart if needed
docker compose restart openwebui
```

For more troubleshooting tips, see [PROJECT_STATUS.md](PROJECT_STATUS.md#troubleshooting).

## 📊 Performance

### Typical Response Times

- **Vector retrieval**: 50-200ms
- **Re-ranking**: 100-500ms
- **LLM generation**: 2-10s
- **Total response**: 3-12s

### Resource Usage

- **Memory**: 4-5 GB total
  - Ollama: ~2 GB
  - PostgreSQL: ~500 MB
  - FastAPI: ~1.5 GB
  - OpenWebUI: ~500 MB
- **CPU**: Moderate (spikes during generation)
- **Disk**: ~5 GB (models + database)

## 🛠️ Development

### Running Tests

```bash
# Health check all endpoints
./test_all_endpoints.sh

# Individual endpoint test
curl http://localhost:4000/health
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f fastapi
docker compose logs -f ollama
docker compose logs -f postgres
```

### Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Anthropic](https://www.anthropic.com/news/contextual-retrieval) - Contextual Retrieval approach
- [Cohere](https://cohere.com/rerank) - Re-ranking models
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [CrewAI](https://www.crewai.com/) - Multi-agent orchestration
- [Arize Phoenix](https://phoenix.arize.com/) - LLM observability
- [OpenWebUI](https://github.com/open-webui/open-webui) - Chat interface

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using LlamaIndex, CrewAI, and OpenWebUI**
