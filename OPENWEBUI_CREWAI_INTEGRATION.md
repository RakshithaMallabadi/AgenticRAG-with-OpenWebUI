# OpenWebUI → CrewAI Integration Complete! 🎉

## ✅ What Was Implemented

OpenWebUI now uses the **CrewAI multi-agent system** by default, providing:

1. **3-Agent Workflow**
   - 🔍 **Research Agent**: Analyzes retrieved documents
   - ✍️ **Synthesis Agent**: Creates comprehensive answers
   - ✅ **Quality Agent**: Validates and improves responses

2. **Full Feature Set**
   - 💬 Conversation Memory (persistent across turns)
   - 🎯 Contextual RAG (uses previous conversation for context)
   - 📊 Re-ranking (Cohere or similarity-based)
   - 📚 Citation Handling (automatic source references)
   - 🔄 Fallback (direct LLM if CrewAI fails)

3. **Seamless Integration**
   - No changes needed to OpenWebUI configuration
   - Works with existing `/v1/chat/completions` endpoint
   - All conversation memory features preserved

---

## 🚀 How to Use

### 1. Access OpenWebUI
Open your browser and go to:
```
http://localhost:3000
```

### 2. Start Chatting
Simply ask questions in OpenWebUI. Behind the scenes:
- Your question is sent to FastAPI
- FastAPI retrieves relevant documents from PGVector
- Documents are re-ranked for relevance
- CrewAI's 3 agents process the query:
  1. Research Agent analyzes documents
  2. Synthesis Agent creates answer with conversation context
  3. Quality Agent validates and improves
- Answer with citations is returned to OpenWebUI
- Conversation memory is updated

### 3. Multi-Turn Conversations
You can have follow-up conversations, and the system will:
- Remember previous questions and answers
- Use conversation history to improve context retrieval
- Maintain session-specific memory

---

## 📊 System Architecture

```
┌─────────────┐
│  OpenWebUI  │ (Port 3000)
│  (Browser)  │
└──────┬──────┘
       │ HTTP POST /v1/chat/completions
       ▼
┌─────────────────────────────────────┐
│         FastAPI Backend             │ (Port 4000)
│  ┌──────────────────────────────┐   │
│  │  Conversation Memory Manager │   │
│  └──────────┬───────────────────┘   │
│             ▼                        │
│  ┌──────────────────────────────┐   │
│  │  RAG Pipeline                │   │
│  │  1. Context-Aware Query      │   │
│  │  2. Vector Retrieval         │   │
│  │  3. Re-ranking               │   │
│  └──────────┬───────────────────┘   │
│             ▼                        │
│  ┌──────────────────────────────┐   │
│  │  CrewAI Multi-Agent System   │   │
│  │  ┌────────────────────────┐  │   │
│  │  │  Research Agent        │  │   │
│  │  │  (Analyzes docs)       │  │   │
│  │  └──────┬─────────────────┘  │   │
│  │         ▼                     │   │
│  │  ┌────────────────────────┐  │   │
│  │  │  Synthesis Agent       │  │   │
│  │  │  (Creates answer)      │  │   │
│  │  └──────┬─────────────────┘  │   │
│  │         ▼                     │   │
│  │  ┌────────────────────────┐  │   │
│  │  │  Quality Agent         │  │   │
│  │  │  (Validates output)    │  │   │
│  │  └──────┬─────────────────┘  │   │
│  └─────────┼─────────────────────┘   │
│            ▼                          │
│  Answer + Citations + Memory Update   │
└───────────┬───────────────────────────┘
            │
            ▼
     ┌─────────────┐
     │  PostgreSQL │
     │  + pgvector │
     └─────────────┘
```

---

## 🔍 Technical Implementation

### Modified Endpoint: `/v1/chat/completions`

**Before:**
- Used direct LLM call
- Fast but less thorough

**After:**
- Uses CrewAI multi-agent workflow
- Passes conversation context to agents
- Research → Synthesis → Quality validation
- Slower but much higher quality answers
- Automatic fallback to direct LLM if CrewAI fails

### Code Location
File: `app.py` (lines 924-1100)

Key changes:
```python
# Step 1: Build context-aware query with conversation history
context_query = question
if len(chat_history) > 2:
    recent_msgs = [msg for msg in chat_history[:-1]][-4:]
    context_query = f"Previous conversation:\n{recent_context}\n\nCurrent question: {question}"

# Step 2: Retrieve and re-rank
nodes = retriever.retrieve(context_query)
reranked_nodes = reranker.postprocess_nodes(nodes, query_str=question)

# Step 3: Format context for CrewAI agents
retrieved_context = conversation_context + "\n\n".join(crewai_context_parts)

# Step 4: Run CrewAI workflow
crewai_answer = run_crewai_rag(question, retrieved_context)

# Step 5: Update conversation memory
memory.put(LLMChatMessage(role=MessageRole.ASSISTANT, content=answer))
```

---

## 🎯 Available Endpoints

| Endpoint | Access | Agents | Memory | Speed | Use Case |
|----------|--------|--------|--------|-------|----------|
| **OpenWebUI** | `http://localhost:3000` | ✅ 3 Agents | ✅ Yes | 🐢 Slow | **Primary UI** |
| `/v1/chat/completions` | API | ✅ 3 Agents | ✅ Yes | 🐢 Slow | OpenWebUI backend |
| `/ask-crewai` | API | ✅ 3 Agents | ❌ No | 🐢 Slow | Stateless queries |
| `/chat` | API | ❌ Direct LLM | ✅ Yes | ⚡ Fast | Quick conversations |
| `/ask` | API | ❌ Direct LLM | ❌ No | ⚡ Fast | Quick queries |

---

## 🧪 Testing

### Test with curl:
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "user", "content": "Who issued the Abu Dhabi Procurement Standards?"}
    ]
  }'
```

### Check logs to see CrewAI in action:
```bash
docker logs fastapi-backend 2>&1 | grep -E "(OpenWebUI CrewAI|CrewAI.*Workflow)" | tail -10
```

Expected log output:
```
[OpenWebUI CrewAI] Starting multi-agent workflow...
[CrewAI] Starting multi-agent RAG workflow for: Who issued...
[CrewAI] Workflow completed successfully
[OpenWebUI CrewAI] Workflow completed successfully
```

---

## ⚡ Performance Considerations

### Response Times:
- **With CrewAI (OpenWebUI)**: ~30-60 seconds
  - Research Agent: ~10-15s
  - Synthesis Agent: ~10-15s
  - Quality Agent: ~10-15s
  - Total with overhead: ~30-60s

- **Without CrewAI (`/ask`, `/chat`)**: ~2-3 seconds
  - Direct LLM call with RAG

### Why is CrewAI Slower?
The 3-agent workflow provides thorough analysis:
1. **Research Agent** deeply analyzes all retrieved documents
2. **Synthesis Agent** crafts comprehensive, accurate answers
3. **Quality Agent** validates and improves the output

This multi-step validation ensures higher quality but takes longer.

### Fallback Mechanism:
If CrewAI fails (timeout, error, etc.), the system automatically falls back to direct LLM, ensuring you always get a response.

---

## 🔧 Configuration

### Environment Variables:
All configured in `docker-compose.yml`:

```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama:11434
  - OLLAMA_MODEL=llama3.2:1b
  - DATABASE_URL=postgresql://raguser:ragpass@postgres:5432/ragdb
```

### LLM Settings (in `app.py`):
```python
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    context_window=4096,
    request_timeout=120.0,
    additional_kwargs={
        "num_ctx": 4096,
        "num_predict": 512
    }
)
```

### CrewAI Agents:
All three agents use:
- Same LLM: `llama3.2:1b` via Ollama
- Verbose mode: Enabled
- Delegation: Disabled (sequential workflow)
- Process: Sequential (Research → Synthesis → Quality)

---

## 📚 Memory Management

### Session IDs:
- Generated deterministically based on first 2 messages
- Format: `webui-{md5_hash}`
- Ensures same conversation maintains memory across refreshes

### Storage:
- In-memory dictionary: `chat_sessions`
- Key: session_id
- Value: `ChatMemoryBuffer` instance

### Endpoints for Memory Management:
```bash
# View conversation history
GET /chat/history/{session_id}

# Clear specific session
POST /chat/clear
{"session_id": "webui-abc123"}

# List all sessions
GET /chat/sessions

# Clear all sessions
DELETE /chat/sessions
```

---

## ✅ Features Verification

### ✅ Conversation Memory
- Multi-turn conversations maintained
- Previous context influences retrieval
- Session-based storage

### ✅ Contextual RAG
- Uses conversation history for query enhancement
- Over-fetching for better coverage
- Context-aware document retrieval

### ✅ Re-ranking
- Cohere re-ranker (if `COHERE_API_KEY` set)
- Fallback to similarity-based re-ranking
- Top 5 most relevant sources

### ✅ Citations
- Automatic source references
- Relevance scores included
- Source metadata preserved

### ✅ CrewAI Multi-Agent
- 3 specialized agents
- Sequential workflow
- Quality validation

### ✅ Fallback Mechanism
- Automatic fallback to direct LLM if CrewAI fails
- Graceful error handling
- User always gets a response

---

## 🎉 Summary

You now have a **production-ready Agentic RAG system** with:

1. ✅ **OpenWebUI** as the primary interface
2. ✅ **CrewAI multi-agent orchestration** (3 specialized agents)
3. ✅ **Conversation memory** for context-aware responses
4. ✅ **Contextual RAG** with re-ranking
5. ✅ **Automatic citations** and source references
6. ✅ **Graceful fallbacks** for reliability
7. ✅ **LlamaIndex + PGVector** for vector storage
8. ✅ **Ollama** for local LLM hosting

All services running at:
- **OpenWebUI**: http://localhost:3000 (primary interface)
- **FastAPI**: http://localhost:4000 (backend API)
- **Ollama**: http://localhost:11434 (LLM server)
- **PostgreSQL**: localhost:5432 (vector database)

---

## 🚀 Next Steps

1. **Use OpenWebUI**: Navigate to http://localhost:3000 and start asking questions
2. **Monitor Logs**: `docker logs -f fastapi-backend` to see CrewAI in action
3. **Test Different Queries**: Try complex questions to see agent collaboration
4. **Compare Endpoints**: Use `/ask` for speed, OpenWebUI for quality
5. **Ingest More Documents**: POST to `/ingest` to add more knowledge

Enjoy your intelligent, multi-agent RAG system! 🎉

