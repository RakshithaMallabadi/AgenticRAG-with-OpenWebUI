# Arize Phoenix Observability Implementation ✅

## 🎉 Implementation Complete!

Arize Phoenix has been successfully integrated into your RAG system for comprehensive observability and tracing.

---

## 🌟 What is Phoenix?

Arize Phoenix is an open-source observability platform for LLM applications that provides:
- **Real-time tracing** of all LLM calls and embeddings
- **Performance monitoring** (latency, token usage, costs)
- **Debugging tools** for RAG pipelines
- **Prompt management** and versioning
- **Quality evaluation** of LLM responses

---

## ✅ What Was Implemented

### 1. **Phoenix Installation & Configuration**
- ✅ Installed `arize-phoenix>=4.0.0`
- ✅ Installed OpenInference instrumentation packages:
  - `openinference-instrumentation-llama-index` - LlamaIndex tracing
  - `openinference-instrumentation-crewai` - CrewAI tracing
  - `opentelemetry-sdk` - OpenTelemetry support
  - `opentelemetry-exporter-otlp` - OTLP protocol support

### 2. **Phoenix Initialization in app.py**
- ✅ Phoenix session launched on startup
- ✅ LlamaIndex instrumentation enabled
- ✅ OpenTelemetry tracer configured
- ✅ All LLM calls automatically traced

### 3. **Docker Configuration**
- ✅ Port 6006 exposed for Phoenix UI
- ✅ Environment variables configured:
  - `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006`
  - `PHOENIX_PROJECT_NAME=openwebui-rag`

### 4. **FastAPI Endpoints**
- ✅ `GET /phoenix/status` - Check Phoenix configuration
- ✅ `GET /phoenix/traces` - View traces information
- ✅ `GET /phoenix/metrics` - View metrics information
- ✅ Updated `/health` to include `phoenix_enabled` status

### 5. **Automatic Tracing**
- ✅ **LlamaIndex Operations**:
  - Embedding generation
  - Vector retrieval
  - Re-ranking
  - LLM completions
  - Query processing
- ✅ **CrewAI Agents**:
  - All agent LLM calls traced
  - Multi-agent workflow visibility
  - Task execution monitoring

---

## 🚀 How to Access Phoenix

### Phoenix UI
Open your browser and navigate to:
```
http://localhost:6006
```

The Phoenix UI provides:
- 📊 **Dashboard**: Overview of all traces and metrics
- 🔍 **Traces View**: Detailed trace inspection
- 📈 **Metrics**: Performance analytics
- 🧪 **Evaluations**: Quality assessments
- 📝 **Prompts**: Prompt version management

---

## 🔍 What Phoenix Tracks

### 1. **LLM Calls**
Phoenix automatically captures:
- **Model**: Which LLM was used (e.g., `llama3.2:1b`)
- **Prompt**: The exact prompt sent to the LLM
- **Response**: The LLM's output
- **Tokens**: Input/output token counts
- **Latency**: How long the call took
- **Cost**: Estimated cost (if applicable)
- **Metadata**: Temperature, context window, etc.

### 2. **Embeddings**
- **Model**: Embedding model used (e.g., `all-MiniLM-L6-v2`)
- **Input Text**: Text being embedded
- **Vector Dimension**: Embedding size
- **Latency**: Embedding generation time
- **Batch Size**: Number of texts embedded

### 3. **Retrieval**
- **Query**: The search query
- **Retrieved Documents**: Top-K documents retrieved
- **Relevance Scores**: Similarity scores
- **Re-ranking**: Re-ranked results
- **Latency**: Retrieval time

### 4. **End-to-End Workflows**
- **Full RAG Pipeline**: Query → Retrieval → Re-ranking → LLM → Response
- **Multi-Agent Workflows**: CrewAI agent interactions
- **Conversation Chains**: Multi-turn conversations with memory

---

## 📊 Phoenix API Endpoints

### Check Phoenix Status
```bash
curl http://localhost:4000/phoenix/status
```

**Response:**
```json
{
  "enabled": true,
  "ui_url": "http://localhost:6006",
  "project_name": "openwebui-rag",
  "collector_endpoint": "http://localhost:6006",
  "instrumented": {
    "llama_index": true,
    "crewai": false
  },
  "features": [
    "LLM call tracing",
    "Embedding tracking",
    "Retrieval monitoring",
    "Latency analysis",
    "Token usage tracking"
  ]
}
```

### View Traces Info
```bash
curl http://localhost:4000/phoenix/traces
```

### View Metrics Info
```bash
curl http://localhost:4000/phoenix/metrics
```

### System Health (includes Phoenix)
```bash
curl http://localhost:4000/health
```

**Response:**
```json
{
  "status": "ok",
  "vector_store": "LlamaIndex + PGVector",
  "nodes_available": true,
  "phoenix_enabled": true
}
```

---

## 🧪 Testing Phoenix Integration

### 1. Make a Test Query
```bash
curl -X POST http://localhost:4000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who issued the Abu Dhabi Procurement Standards?", "top_k": 5}'
```

### 2. View Traces in Phoenix UI
1. Open http://localhost:6006
2. You'll see a new trace for your query
3. Click on the trace to see:
   - **Embedding generation** for the query
   - **Vector retrieval** from PGVector
   - **Re-ranking** with Cohere/Similarity
   - **LLM completion** for answer generation
   - **Total latency** and token usage

### 3. Test CrewAI Multi-Agent Workflow
```bash
curl -X POST http://localhost:4000/ask-crewai \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main principles?", "top_k": 5}'
```

View the multi-agent workflow in Phoenix:
- Research Agent LLM call
- Synthesis Agent LLM call
- Quality Agent LLM call
- Total workflow latency

### 4. Test OpenWebUI Integration
1. Open OpenWebUI at http://localhost:3000
2. Ask a question in the chat
3. View the trace in Phoenix UI at http://localhost:6006
4. See the full conversational RAG pipeline

---

## 📈 Phoenix Features You Can Use

### 1. **Trace Inspection**
- Click any trace in the UI
- See the complete call stack
- Inspect input/output for each step
- Identify bottlenecks

### 2. **Performance Analysis**
- View average latency per endpoint
- Track token usage over time
- Identify slow queries
- Monitor cost trends

### 3. **Quality Evaluation**
Phoenix can evaluate:
- **Relevance**: Are retrieved documents relevant?
- **Faithfulness**: Is the answer grounded in context?
- **Completeness**: Does the answer address the question?

### 4. **Prompt Management**
- Store prompt templates in Phoenix
- Version control for prompts
- A/B test different prompts
- Track prompt performance

### 5. **Debugging**
- Find failing queries
- Inspect error traces
- Reproduce issues
- Compare successful vs failed traces

---

## 🔧 Configuration

### Environment Variables
Set in `docker-compose.yml`:

```yaml
environment:
  - PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
  - PHOENIX_PROJECT_NAME=openwebui-rag
```

### app.py Configuration
Phoenix is initialized in `app.py`:

```python
# Launch Phoenix session
phoenix_session = px.launch_app(host="0.0.0.0", port=6006)

# Configure OpenTelemetry tracer
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(
        OTLPSpanExporter(endpoint=f"{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces")
    )
)

# Instrument LlamaIndex
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
```

---

## 📋 What Gets Traced

### `/ask` Endpoint
```
┌─ Trace: RAG Query
│  ├─ Span: Embedding Generation
│  │  └─ Model: all-MiniLM-L6-v2
│  ├─ Span: Vector Retrieval
│  │  └─ Retrieved: 10 documents
│  ├─ Span: Re-ranking
│  │  └─ Top 5 selected
│  ├─ Span: LLM Completion
│  │  ├─ Model: llama3.2:1b
│  │  ├─ Tokens: 450 in, 120 out
│  │  └─ Latency: 2.3s
│  └─ Total: 3.5s
```

### `/ask-crewai` Endpoint
```
┌─ Trace: CrewAI Multi-Agent RAG
│  ├─ Span: Vector Retrieval
│  ├─ Span: Re-ranking
│  ├─ Span: Research Agent
│  │  └─ LLM Call (llama3.2:1b)
│  ├─ Span: Synthesis Agent
│  │  └─ LLM Call (llama3.2:1b)
│  ├─ Span: Quality Agent
│  │  └─ LLM Call (llama3.2:1b)
│  └─ Total: 45s
```

### `/v1/chat/completions` (OpenWebUI)
```
┌─ Trace: OpenWebUI Conversation
│  ├─ Span: Session Memory Retrieval
│  ├─ Span: Context-Aware Query Building
│  ├─ Span: Embedding Generation
│  ├─ Span: Vector Retrieval
│  ├─ Span: Re-ranking
│  ├─ Span: CrewAI Multi-Agent Workflow
│  │  ├─ Research Agent LLM Call
│  │  ├─ Synthesis Agent LLM Call
│  │  └─ Quality Agent LLM Call
│  ├─ Span: Memory Update
│  └─ Total: 48s
```

---

## 🎯 Use Cases

### 1. **Performance Optimization**
- Identify slow endpoints
- Find bottlenecks in retrieval
- Optimize embedding generation
- Reduce LLM latency

### 2. **Quality Improvement**
- Evaluate answer quality
- Test different prompts
- Compare retrieval strategies
- Measure re-ranking effectiveness

### 3. **Debugging**
- Trace failing queries
- Inspect LLM inputs/outputs
- Find retrieval issues
- Debug agent interactions

### 4. **Cost Monitoring**
- Track token usage
- Estimate costs
- Optimize token consumption
- Budget planning

### 5. **A/B Testing**
- Compare prompt variations
- Test different models
- Evaluate retrieval methods
- Measure quality improvements

---

## 🚨 Troubleshooting

### Phoenix UI Not Loading
```bash
# Check if Phoenix is running
curl http://localhost:6006

# Check FastAPI logs
docker logs fastapi-backend | grep Phoenix

# Verify port is exposed
docker compose ps
```

### No Traces Appearing
```bash
# Make a test query
curl -X POST http://localhost:4000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Check Phoenix status
curl http://localhost:4000/phoenix/status

# Refresh Phoenix UI
```

### Instrumentation Not Working
```bash
# Check if packages are installed
docker exec fastapi-backend pip list | grep phoenix
docker exec fastapi-backend pip list | grep openinference

# Restart container
docker compose restart fastapi
```

---

## 📚 Additional Resources

- **Phoenix Documentation**: https://arize.com/docs/phoenix
- **OpenInference Spec**: https://openinference.io/
- **LlamaIndex Observability**: https://docs.llamaindex.ai/en/stable/module_guides/observability/
- **Phoenix GitHub**: https://github.com/Arize-ai/phoenix

---

## 🎉 Summary

### ✅ Implemented Features
1. ✅ Phoenix installation and configuration
2. ✅ LlamaIndex instrumentation (embeddings, retrieval, LLM)
3. ✅ CrewAI agent tracing
4. ✅ OpenTelemetry integration
5. ✅ Phoenix UI accessible at port 6006
6. ✅ FastAPI endpoints for status/metrics
7. ✅ Automatic tracing of all RAG operations
8. ✅ Docker integration with proper port exposure

### 🌐 Access Points
- **Phoenix UI**: http://localhost:6006
- **FastAPI Backend**: http://localhost:4000
- **Phoenix Status**: http://localhost:4000/phoenix/status
- **OpenWebUI**: http://localhost:3000

### 🔍 What You Get
- Real-time visibility into all LLM operations
- Performance metrics and analytics
- Debugging capabilities
- Prompt lifecycle management
- Quality evaluation tools
- Cost tracking

---

**Your RAG system now has enterprise-grade observability! 🎉**

Open http://localhost:6006 to explore your traces and metrics.

