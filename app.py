from __future__ import annotations

# ---- macOS / TF guards (keep immediately after the future import) ----
import os as _os

_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# ----------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Dict
import json
import hashlib

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import httpx

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter

# Re-ranking and Contextual RAG
from llama_index.core.postprocessor import SimilarityPostprocessor
try:
    from llama_index.postprocessor.cohere_rerank import CohereRerank
    _HAS_COHERE = True
except ImportError:
    _HAS_COHERE = False
    print("[Rerank] Cohere reranker not available, using similarity-based reranking")

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

# Memory and chat
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage as LLMChatMessage, MessageRole
import uuid

# CrewAI
from crewai import Agent, Task, Crew, Process

# Arize Phoenix - Observability and Tracing
try:
    import phoenix as px
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    _HAS_PHOENIX = True
    print("[Phoenix] Phoenix observability loaded successfully")
except ImportError as e:
    _HAS_PHOENIX = False
    print(f"[Phoenix] Phoenix not available: {e}")

# Optional Docling support (PDF parsing)
try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except Exception as e:
    print(f"[Docling] Failed to import: {e}")
    DocumentConverter = None
    _HAS_DOCLING = False

# RAGAs - Evaluation Framework
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from datasets import Dataset
    _HAS_RAGAS = True
    print("[RAGAs] RAGAs evaluation framework loaded successfully")
except ImportError as e:
    _HAS_RAGAS = False
    print(f"[RAGAs] RAGAs not available: {e}")

load_dotenv()

# ---------- Paths ----------
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"

# ---------- LlamaIndex Configuration ----------

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://raguser:ragpass@postgres:5432/ragdb"
)

# Configure embedding model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Configure LLM
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE,
    temperature=0.2,
    context_window=4096,  # Increased for better accuracy
    request_timeout=120.0,
    additional_kwargs={
        "num_ctx": 4096,  # Context window size
        "num_predict": 512,  # Max tokens to generate
    },
)

# Set global LlamaIndex settings
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Initialize PGVector store
print("[LlamaIndex] Initializing PGVectorStore...")
vector_store = PGVectorStore.from_params(
    database=DATABASE_URL.split("/")[-1],
    host=DATABASE_URL.split("@")[1].split(":")[0],
    password=DATABASE_URL.split(":")[2].split("@")[0],
    port=int(DATABASE_URL.split(":")[-1].split("/")[0]),
    user=DATABASE_URL.split("://")[1].split(":")[0],
    table_name="llamaindex_documents",
    embed_dim=384,  # dimension for all-MiniLM-L6-v2
)

# Create storage context and index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)
print("[LlamaIndex] Vector store initialized successfully")

# ---------- Arize Phoenix Initialization ----------

phoenix_session = None
if _HAS_PHOENIX:
    try:
        # Get Phoenix configuration from environment
        PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")
        PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "openwebui-rag")
        
        print(f"[Phoenix] Starting Phoenix session for project: {PHOENIX_PROJECT_NAME}")
        print(f"[Phoenix] Collector endpoint: {PHOENIX_COLLECTOR_ENDPOINT}")
        
        # Launch Phoenix session
        phoenix_session = px.launch_app(host="0.0.0.0", port=6006)
        
        # Configure OpenTelemetry tracer to send to Phoenix
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(
                OTLPSpanExporter(endpoint=f"{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces")
            )
        )
        
        # Instrument LlamaIndex with Phoenix tracing
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] ✅ LlamaIndex instrumentation enabled")
        
        # Instrument CrewAI with Phoenix tracing
        try:
            from openinference.instrumentation.crewai import CrewAIInstrumentor
            CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
            print("[Phoenix] ✅ CrewAI instrumentation enabled")
        except Exception as e:
            print(f"[Phoenix] ⚠️  CrewAI instrumentation failed: {e}")
        
        print("[Phoenix] ✅ Phoenix tracing enabled")
        print(f"[Phoenix] 🌐 Phoenix UI available at: http://localhost:6006")
        
    except Exception as e:
        print(f"[Phoenix] ⚠️  Failed to initialize Phoenix: {e}")
        print("[Phoenix] Continuing without observability...")
        _HAS_PHOENIX = False
else:
    print("[Phoenix] Phoenix not available - install with: pip install arize-phoenix openinference-instrumentation-llama-index")

# ---------- Contextual Retrieval & Re-ranking Configuration ----------

# Contextual prompt template (Anthropic-style)
CONTEXTUAL_PROMPT = PromptTemplate(
    """You are an AI assistant tasked with providing specific context to passages for better retrieval.
    
    Here is the document context:
    {context}
    
    Here is the chunk we want to situate within the whole document:
    {chunk}
    
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    """
)

# Initialize re-ranker (Cohere or similarity-based fallback)
if _HAS_COHERE and os.getenv("COHERE_API_KEY"):
    print("[Rerank] Using Cohere re-ranker")
    reranker = CohereRerank(
        api_key=os.getenv("COHERE_API_KEY"),
        top_n=5,
        model="rerank-english-v3.0"
    )
else:
    print("[Rerank] Using similarity-based re-ranker")
    reranker = SimilarityPostprocessor(similarity_cutoff=0.1)  # Lowered from 0.3 to allow more results

# Response synthesizer for contextual answers
response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    use_async=False,
)

# ---------- RAGAs Configuration ----------

ragas_llm = None
ragas_embeddings = None

if _HAS_RAGAS:
    try:
        # Wrap LlamaIndex LLM and embeddings for RAGAs
        ragas_llm = LlamaIndexLLMWrapper(llm)
        ragas_embeddings = LlamaIndexEmbeddingsWrapper(embed_model)
        print("[RAGAs] RAGAs evaluators configured successfully")
    except Exception as e:
        print(f"[RAGAs] Failed to configure RAGAs: {e}")
        _HAS_RAGAS = False
else:
    print("[RAGAs] RAGAs not available - install with: pip install ragas datasets")

# ---------- Conversation Memory ----------

# In-memory session storage (use Redis in production for persistence)
chat_sessions: Dict[str, ChatMemoryBuffer] = {}

def get_or_create_memory(session_id: str) -> ChatMemoryBuffer:
    """Get existing chat memory or create a new one for the session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatMemoryBuffer.from_defaults(
            token_limit=3000,  # Adjust based on your model's context window
        )
        print(f"[Memory] Created new session: {session_id}")
    else:
        print(f"[Memory] Using existing session: {session_id}")
    return chat_sessions[session_id]

def clear_session(session_id: str) -> bool:
    """Clear conversation history for a session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        print(f"[Memory] Cleared session: {session_id}")
        return True
    return False

# ---------- Docling Helper ----------
_DOC_CONVERTER: DocumentConverter | None = None

def _get_docling() -> DocumentConverter | None:
    if not _HAS_DOCLING:
        print("[Docling] Docling package not available")
        return None
    global _DOC_CONVERTER
    if _DOC_CONVERTER is None:
        try:
            _DOC_CONVERTER = DocumentConverter()
            print("[Docling] DocumentConverter initialized successfully")
        except Exception as e:
            print(f"[Docling] Failed to initialize DocumentConverter: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _DOC_CONVERTER

def _read_text_from_path(p: Path) -> str:
    """
    Returns text for .txt/.md; uses Docling for .pdf.
    """
    suf = p.suffix.lower()
    if suf in {".txt", ".md"}:
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[ingest] plain read failed {p}: {e}")
            return ""

    if suf == ".pdf":
        converter = _get_docling()
        if not converter:
            print(f"[Docling] Docling not available, skipping {p}")
            return ""
            
        try:
            result = None
            try:
                result = converter.convert(str(p))
            except Exception as e1:
                print(f"[Docling] Direct convert failed for {p}: {e1}")
                return ""

            doc = getattr(result, "document", None) if result else None
            if doc is None:
                print(f"[Docling] No document in result for {p}")
                return ""

            # Try markdown export
            try:
                if hasattr(doc, "export_to_markdown"):
                    md = doc.export_to_markdown()
                    if md and md.strip():
                        print(f"[Docling] Parsed {p.name}: markdown length={len(md)}")
                        return md
            except Exception as e:
                print(f"[Docling] export_to_markdown failed: {e}")

            # Fallback to text
            try:
                if hasattr(doc, "export_to_text"):
                    txt = doc.export_to_text()
                    if txt and txt.strip():
                        print(f"[Docling] Parsed {p.name}: text length={len(txt)}")
                        return txt
            except Exception as e:
                print(f"[Docling] export_to_text failed: {e}")

            print(f"[Docling] No export method succeeded for {p}")
            return ""
        except Exception as e:
            print(f"[Docling] Failed to parse {p}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    return ""

# ---------- Contextual Enhancement Function ----------
def add_contextual_metadata(documents: List[Document]) -> List[Document]:
    """
    Add contextual information to each chunk (Anthropic-style contextual retrieval)
    This enriches each chunk with document-level context for better retrieval.
    """
    contextual_docs = []
    
    for doc in documents:
        # Get document-level context (first 500 chars or summary)
        doc_context = doc.text[:500] if len(doc.text) > 500 else doc.text
        
        # Add contextual prefix to make retrieval more accurate
        source_name = doc.metadata.get("source", "unknown")
        file_type = doc.metadata.get("file_type", "")
        
        # Create contextual prefix
        contextual_prefix = f"Document: {source_name} (Type: {file_type})\n"
        contextual_prefix += f"Context: This content is from {source_name}. "
        
        # Add to metadata for retrieval
        doc.metadata["contextual_info"] = contextual_prefix
        doc.metadata["doc_context"] = doc_context
        
        contextual_docs.append(doc)
    
    return contextual_docs

# ---------- Ingestion Function ----------
def ingest_documents(folder: Path = DATA_DIR, patterns: tuple = ("*.txt", "*.md", "*.pdf")):
    """Ingest documents using LlamaIndex with contextual enhancement"""
    folder = Path(folder).resolve()
    files = []
    for pat in patterns:
        files.extend(sorted(folder.rglob(pat)))

    documents = []
    for p in files:
        try:
            text = _read_text_from_path(p)
            if not text or not text.strip():
                continue
            
            # Create LlamaIndex Document with metadata
            doc = Document(
                text=text,
                metadata={
                    "source": p.name,
                    "file_path": str(p),
                    "file_type": p.suffix,
                }
            )
            documents.append(doc)
            print(f"[Ingest] Added document: {p.name}")
        except Exception as e:
            print(f"[ingest] skip {p}: {e}")
            continue

    if not documents:
        return 0, [str(p) for p in files]

    # Add contextual metadata (Anthropic-style)
    print(f"[Contextual RAG] Adding contextual metadata to {len(documents)} documents")
    documents = add_contextual_metadata(documents)
    
    # Parse documents into nodes with chunking
    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    nodes = text_splitter.get_nodes_from_documents(documents)
    
    # Enhance nodes with contextual information
    for node in nodes:
        if "contextual_info" in node.metadata:
            # Prepend contextual info to node text for better retrieval
            node.text = f"{node.metadata['contextual_info']}\n{node.text}"
    
    print(f"[LlamaIndex] Created {len(nodes)} contextual nodes from {len(documents)} documents")
    
    # Insert nodes into vector store
    index.insert_nodes(nodes)
    
    print(f"[LlamaIndex] Ingested {len(nodes)} contextually-enhanced chunks")
    return len(nodes), [str(p) for p in files]

# ---------- CrewAI Agents ----------
# ---------- CrewAI Agents ----------
# Set environment variables for CrewAI
os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE)

# Initialize CrewAI agents only if LLM is available
rag_agent = None
if llm is not None:
    try:
        # Legacy single agent (kept for compatibility)
        rag_agent = Agent(
            role="RAG Answerer",
            goal=(
                "Answer the user's question using ONLY the provided context chunks. "
                "Cite sources as [filename#chunk-N]. If context is insufficient, say so explicitly."
            ),
            backstory=(
                "You are a careful analyst. You never invent facts. You keep answers concise "
                "and always include exact sources from the provided context."
            ),
            llm=llm,
            verbose=False,
        )
        print("[CrewAI] Single agent initialized successfully")
    except Exception as e:
        print(f"[CrewAI] Failed to initialize single agent: {e}")
        rag_agent = None
else:
    print("[CrewAI] LLM not available, skipping agent initialization")


# ---------- Multi-Agent CrewAI System ----------

def create_rag_crew(question: str, retrieved_context: str) -> Crew:
    """
    Create a CrewAI crew for advanced RAG question answering.
    
    This creates a three-agent system:
    1. Research Agent - Analyzes and extracts information from documents
    2. Synthesis Agent - Creates comprehensive, accurate answers
    3. Quality Agent - Validates answer quality and completeness
    """
    
    # Research Agent - Analyzes retrieved documents
    research_agent = Agent(
        role='Document Research Specialist',
        goal='Analyze retrieved documents and extract all relevant information including names, dates, organizations, and facts',
        backstory="""You are an expert document analyst with exceptional attention to detail. 
        You excel at identifying and extracting specific information like full names (with titles), 
        organizations, dates, numbers, and key relationships from source documents. You never 
        generalize when specific details are available.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    # Synthesis Agent - Creates comprehensive answers
    synthesis_agent = Agent(
        role='Information Synthesis Expert',
        goal='Synthesize extracted information into clear, accurate, and well-cited answers',
        backstory="""You are a skilled communicator who transforms analyzed data into clear, 
        concise answers. You always include specific details (names with titles, complete 
        organization names, exact dates) and cite sources properly. You are faithful to the 
        source material and never add information not present in the documents.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    # Quality Agent - Validates answers
    quality_agent = Agent(
        role='Quality Assurance Specialist',
        goal='Ensure answers are accurate, complete, well-cited, and include all necessary specific details',
        backstory="""You are meticulous about quality and accuracy. You verify that answers 
        include all key details (full names with titles, complete organization names, specific 
        dates), proper citations, and are completely faithful to source documents. You identify 
        any missing information or areas needing improvement.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    # Task 1: Research and extract information
    research_task = Task(
        description=f"""Analyze the following retrieved documents to answer: "{question}"

Retrieved Context:
{retrieved_context}

Your task is to extract ALL relevant information including:
- Full names with titles (e.g., H.E., Dr., Chairman, etc.)
- Complete organization names and departments
- Specific dates, version numbers, and other numerical data
- Key facts, roles, and relationships
- Direct quotes if helpful

Provide a structured analysis with all extracted details.""",
        agent=research_agent,
        expected_output="Detailed structured analysis of all relevant information from the source documents with specific names, titles, organizations, and dates"
    )
    
    # Task 2: Synthesize into answer
    synthesis_task = Task(
        description=f"""Using the research analysis, create a comprehensive answer to: "{question}"

Your answer MUST:
- Include specific details: full names WITH titles, complete organization names, exact dates
- Be clear and well-structured
- Cite sources appropriately
- Be completely faithful to the source documents
- Never generalize when specific details exist

Use the research findings to craft your response with precision.""",
        agent=synthesis_agent,
        expected_output="Clear, precise answer with all specific details (names with titles, organizations, dates) and proper source citations",
        context=[research_task]
    )
    
    # Task 3: Quality check and improvement
    quality_task = Task(
        description="""Review the synthesized answer for:

1. **Accuracy**: Is it faithful to source documents? No invented facts?
2. **Completeness**: Includes all key details (full names with titles, complete org names, exact dates)?
3. **Specificity**: Are specific details used instead of generalizations?
4. **Citations**: Are sources properly referenced?
5. **Clarity**: Is it easy to understand?

If the answer is missing specific details that were in the research (like full names, titles, or organizations), 
ADD them to improve the answer. Otherwise, approve and return the final polished answer.""",
        agent=quality_agent,
        expected_output="Final quality-checked and improved answer with all specific details, proper citations, and verified accuracy",
        context=[synthesis_task]
    )
    
    # Create and return the crew with sequential process
    crew = Crew(
        agents=[research_agent, synthesis_agent, quality_agent],
        tasks=[research_task, synthesis_task, quality_task],
        process=Process.sequential,
        verbose=True,
    )
    
    return crew


def run_crewai_rag(question: str, retrieved_context: str) -> str:
    """
    Run the CrewAI multi-agent RAG workflow and return the final answer.
    
    This orchestrates three agents working sequentially to produce
    high-quality, validated answers with proper citations.
    
    Phoenix tracing is automatically enabled for all LLM calls within CrewAI agents.
    """
    try:
        print(f"[CrewAI] Starting multi-agent RAG workflow for: {question}")
        
        # Phoenix automatically traces LLM calls made by CrewAI agents
        # through the LlamaIndex instrumentation
        crew = create_rag_crew(question, retrieved_context)
        result = crew.kickoff()
        
        print(f"[CrewAI] Workflow completed successfully")
        if _HAS_PHOENIX:
            print(f"[Phoenix] CrewAI traces available at http://localhost:6006")
        
        return str(result)
    except Exception as e:
        print(f"[CrewAI] Error in workflow: {e}")
        if _HAS_PHOENIX:
            print(f"[Phoenix] Error traces available at http://localhost:6006")
        raise

# ---------- FastAPI ----------
app = FastAPI(title="LlamaIndex RAG (FastAPI + CrewAI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models ----
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool | None = False
    temperature: float | None = 0.2

class AskRequest(BaseModel):
    question: str
    top_k: int | None = 5
    session_id: str | None = None  # Optional session ID for conversation tracking

class ChatHistoryRequest(BaseModel):
    session_id: str

class ClearHistoryRequest(BaseModel):
    session_id: str

class EvaluateRequest(BaseModel):
    """Request model for RAGAs batch evaluation"""
    questions: List[str]
    ground_truths: List[str] | None = None  # Optional reference answers
    contexts: List[List[str]] | None = None  # Optional pre-retrieved contexts
    top_k: int | None = 5

# ---------- RAGAs Evaluation Helper Functions ----------

def evaluate_rag_response(
    question: str, 
    answer: str, 
    contexts: List[str],
    ground_truth: str | None = None
) -> Dict[str, float]:
    """
    Evaluate a single RAG response using RAGAs metrics.
    
    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context strings
        ground_truth: Optional reference answer for comparison
    
    Returns:
        Dictionary of metric scores (0-1, higher is better)
    """
    if not _HAS_RAGAS:
        return {"error": "RAGAs not available"}
    
    try:
        # Create dataset for evaluation
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        # Add ground truth if provided
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on available data
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]
        
        # Add context_recall only if ground_truth is provided
        if ground_truth:
            metrics.append(context_recall)
        
        print(f"[RAGAs] Running evaluation with {len(metrics)} metrics...")
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        
        # Convert to dict with float values
        scores = {}
        for metric_name, metric_value in result.items():
            if isinstance(metric_value, (list, tuple)) and len(metric_value) > 0:
                scores[metric_name] = float(metric_value[0])
            elif isinstance(metric_value, (int, float)):
                scores[metric_name] = float(metric_value)
        
        print(f"[RAGAs] Evaluation complete: {scores}")
        return scores
        
    except Exception as e:
        print(f"[RAGAs] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# ---- Endpoints ----
@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "data_dir": str(DATA_DIR),
        "vector_store": "LlamaIndex + PGVector",
        "features": {
            "conversation_memory": True,
            "contextual_rag": True,
            "reranking": True,
            "citations": True,
            "crewai_multi_agent": True,
            "document_processing": "Docling",
            "phoenix_observability": _HAS_PHOENIX,
            "ragas_evaluation": _HAS_RAGAS,
        },
        "endpoints": {
            "ask": "POST /ask - Stateless RAG queries with re-ranking",
            "ask_crewai": "POST /ask-crewai - Multi-agent CrewAI RAG (research + synthesis + quality validation)",
            "chat": "POST /chat - Conversational RAG with memory",
            "chat_history": "GET /chat/history/{session_id} - View conversation history",
            "chat_clear": "POST /chat/clear - Clear specific session history",
            "chat_sessions": "GET /chat/sessions - List all active sessions",
            "ingest": "POST /ingest - Ingest documents into vector store",
            "health": "GET /health - System health check",
            "phoenix_status": "GET /phoenix/status - Phoenix observability status",
            "phoenix_traces": "GET /phoenix/traces - View traces information",
            "phoenix_metrics": "GET /phoenix/metrics - View metrics information",
            "ragas_status": "GET /ragas/status - RAGAs evaluation status",
            "evaluate": "POST /evaluate - Evaluate single query with RAGAs metrics",
            "evaluate_batch": "POST /evaluate/batch - Batch evaluation with RAGAs"
        },
        "active_sessions": len(chat_sessions),
        "crewai": {
            "enabled": True,
            "agents": ["research_specialist", "synthesis_expert", "quality_assurance"],
            "workflow": "sequential"
        },
        "phoenix": {
            "enabled": _HAS_PHOENIX,
            "ui_url": "http://localhost:6006" if _HAS_PHOENIX else None,
            "features": ["LLM tracing", "Embedding tracking", "Retrieval monitoring"] if _HAS_PHOENIX else []
        },
        "ragas": {
            "enabled": _HAS_RAGAS,
            "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if _HAS_RAGAS else [],
            "status": "operational" if _HAS_RAGAS and ragas_llm and ragas_embeddings else "not_configured"
        }
    }

@app.post("/ingest")
def ingest(folder: str = Query(default=str(DATA_DIR), description="Folder to ingest")):
    n, files = ingest_documents(folder)
    return {
        "folder": str(Path(folder).resolve()),
        "files_seen": files,
        "ingested_chunks": n,
    }

@app.post("/chat")
def chat_with_memory(body: AskRequest):
    """
    Conversational RAG endpoint with memory.
    
    Maintains conversation history across multiple messages.
    Use session_id to track conversations. If not provided, creates a new session.
    
    Pipeline:
    1. Get or create chat memory for the session
    2. Build context-aware query using conversation history
    3. Retrieve and re-rank relevant documents
    4. Format context with citations
    5. Generate conversational answer
    6. Update conversation memory
    """
    print(f"[Chat] Received question: {body.question}")
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    
    # Get or create session
    session_id = body.session_id or str(uuid.uuid4())
    memory = get_or_create_memory(session_id)
    
    # Step 1: Build context-aware query using chat history
    chat_history = memory.get_all()
    context_query = question
    
    if chat_history:
        # Append recent conversation context to improve retrieval
        recent_msgs = chat_history[-4:] if len(chat_history) > 4 else chat_history
        recent_context = "\n".join([
            f"{msg.role}: {msg.content}" 
            for msg in recent_msgs
        ])
        context_query = f"Previous context:\n{recent_context}\n\nCurrent question: {question}"
        print(f"[Chat] Using conversation context from {len(chat_history)} previous messages")
    else:
        print(f"[Chat] Starting new conversation for session {session_id}")
    
    # Step 2: Retrieve initial candidates (over-fetch for re-ranking)
    initial_top_k = (body.top_k or 5) * 2
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=initial_top_k,
    )
    
    nodes = retriever.retrieve(context_query)
    print(f"[Chat] Retrieved {len(nodes)} initial candidates")
    
    # Step 3: Re-rank the nodes
    try:
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_str=question,  # Use original question for re-ranking
        )
        print(f"[Chat] Re-ranked to {len(reranked_nodes)} nodes")
    except Exception as e:
        print(f"[Chat] Re-ranking failed: {e}, using original nodes")
        reranked_nodes = nodes[:body.top_k or 5]
    
    # Step 4: Format context with citations
    context_parts = []
    citations = []
    
    # Filter nodes by minimum relevance threshold
    min_relevance_threshold = 0.25  # Only include nodes with relevance > 0.25 (balanced threshold)
    filtered_nodes = []
    
    for i, node in enumerate(reranked_nodes[:body.top_k or 5]):
        source = node.metadata.get("source", "unknown")
        file_path = node.metadata.get("file_path", "")
        score = node.score if hasattr(node, 'score') else 0.0
        
        # Only include nodes above relevance threshold
        if score >= min_relevance_threshold:
            filtered_nodes.append(node)
            
            # Create citation
            citation = {
                "index": len(citations) + 1,
                "source": source,
                "file_path": file_path,
                "score": float(score),
                "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
            }
            citations.append(citation)
            
            # Format context part
            context_parts.append(
                f"[{len(citations)}] Source: {source} (Score: {score:.3f})\n{node.text}\n"
            )
    
    context_text = "\n\n".join(context_parts) if context_parts else "No documents found."
    
    # Check if we have sufficient relevant content
    if not filtered_nodes:
        answer = "I don't have sufficient relevant information in the knowledge base to answer this question."
        # Update memory with the "no information" response
        memory.put(LLMChatMessage(role=MessageRole.USER, content=question))
        memory.put(LLMChatMessage(role=MessageRole.ASSISTANT, content=answer))
        
        return {
            "answer": answer,
            "context": "No relevant documents found above the relevance threshold.",
            "retrieved_nodes": 0,
            "citations": [],
            "reranked": True,
            "reranker_type": "cohere" if _HAS_COHERE and os.getenv("COHERE_API_KEY") else "similarity",
            "session_id": session_id,
            "conversation_length": len(memory.get_all()),
            "is_new_session": len(chat_history) == 0,
            "confidence": "low"
        }
    
    # Step 5: Generate conversational answer using LLM
    if context_parts:
        # Build a concise context for LLM (show more text for better accuracy)
        simple_context = "\n\n".join([
            f"Source {i+1}: {node.text[:1000]}" 
            for i, node in enumerate(reranked_nodes[:body.top_k or 5])
        ])
        
        # Add conversation context if exists
        conversation_context = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            conversation_context = "Previous conversation:\n" + "\n".join([
                f"{msg.role}: {msg.content[:150]}" 
                for msg in recent_history
            ]) + "\n\n"
        
        prompt = f"""{conversation_context}Context from documents:
{simple_context}

Question: {question}

Please provide a clear, accurate answer based ONLY on the information in the context above. Extract specific details like names, organizations, dates, and titles exactly as they appear in the documents. Reference source numbers when appropriate."""
        
        # Generate answer using LLM
        try:
            from llama_index.core.llms import ChatMessage as LLMChatMsg
            
            response = llm.chat([
                LLMChatMsg(role="system", content="""You are a precise document analyst. RULES:
• Answer the question using ONLY the information explicitly present in the provided context
• Extract and present specific details like names, dates, terms, and definitions
• If the EXACT terms aren't found but RELATED information exists, explain what IS available
• Only say "I don't have sufficient information" if NO related information is found
• Be helpful - if you find relevant information, share it even if it uses different terminology
• Include full names with titles, organization names, dates, numbers, and roles when present
• Quote directly from source when possible
• NEVER make up information not in the context"""),
                LLMChatMsg(role="user", content=prompt)
            ])
            
            llm_answer = str(response.message.content)
            
            # Validate response - only block if clearly stating no information at the start
            if llm_answer.strip().startswith("I don't have sufficient information"):
                # If LLM clearly says no information, don't add sources
                answer = "I don't have sufficient information in the provided documents to answer this question."
            else:
                # Add source citations for all other responses
                citations_text = "\n\n**Sources:**\n" + "\n".join([
                    f"[{i+1}] {node.metadata.get('source', 'unknown')} (relevance: {node.score if hasattr(node, 'score') else 0.0:.2f})"
                    for i, node in enumerate(filtered_nodes)
                ])
                answer = llm_answer + citations_text
            
        except Exception as e:
            print(f"[Chat] LLM generation failed: {e}, returning context-based answer")
            # Fallback to context-based answer
            answer = f"""Based on the retrieved documents:

{context_text}
---
Retrieved {len(filtered_nodes)} contextually-enhanced chunks."""
    else:
        answer = "No relevant information found in the knowledge base for your question."
    
    # Step 6: Update conversation memory
    memory.put(LLMChatMessage(role=MessageRole.USER, content=question))
    memory.put(LLMChatMessage(role=MessageRole.ASSISTANT, content=answer))
    print(f"[Chat] Updated memory. Total messages: {len(memory.get_all())}")
    
    return {
        "answer": answer,
        "context": context_text,
        "retrieved_nodes": len(filtered_nodes),
        "citations": citations,
        "reranked": True,
        "reranker_type": "cohere" if _HAS_COHERE and os.getenv("COHERE_API_KEY") else "similarity",
        "session_id": session_id,
        "conversation_length": len(memory.get_all()),
        "is_new_session": len(chat_history) == 0,
        "confidence": "high" if len(filtered_nodes) >= 2 else "medium"
    }

@app.post("/ask")
def ask(body: AskRequest):
    """
    Contextual Agentic RAG endpoint with re-ranking.
    
    Pipeline:
    1. Retrieve initial candidates using vector similarity
    2. Re-rank using Cohere or similarity-based reranker
    3. Format with citations and metadata
    4. Return contextually-enriched answer
    """
    print(f"[Contextual RAG] Received question: {body.question}")
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    
    # Check if question is asking for specific terms/concepts that might not exist
    specific_terms = ["delivery terms", "payment terms", "intellectual property", "disciplinary actions", "penalty clauses"]
    question_lower = question.lower()
    asking_for_specific_terms = any(term in question_lower for term in specific_terms)

    # Step 1: Retrieve initial candidates (over-fetch for re-ranking)
    initial_top_k = (body.top_k or 5) * 2  # Fetch 2x for re-ranking
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=initial_top_k,
    )
    
    # Retrieve nodes
    nodes = retriever.retrieve(question)
    print(f"[Contextual RAG] Retrieved {len(nodes)} initial candidates")
    
    # Step 2: Re-rank the nodes
    try:
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_str=question,
        )
        print(f"[Rerank] Re-ranked to {len(reranked_nodes)} nodes")
    except Exception as e:
        print(f"[Rerank] Re-ranking failed: {e}, using original nodes")
        reranked_nodes = nodes[:body.top_k or 5]
    
    # Step 3: Format context with citations and metadata
    context_parts = []
    citations = []
    
    # Filter nodes by minimum relevance threshold
    min_relevance_threshold = 0.25  # Only include nodes with relevance > 0.25 (balanced threshold)
    filtered_nodes = []
    
    for i, node in enumerate(reranked_nodes[:body.top_k or 5]):
        source = node.metadata.get("source", "unknown")
        file_path = node.metadata.get("file_path", "")
        score = node.score if hasattr(node, 'score') else 0.0
        
        # Only include nodes above relevance threshold
        if score >= min_relevance_threshold:
            filtered_nodes.append(node)
            
            # Create citation
            citation = {
                "index": len(citations) + 1,
                "source": source,
                "file_path": file_path,
                "score": float(score),
                "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
            }
            citations.append(citation)
            
            # Format context part
            context_parts.append(
                f"[{len(citations)}] Source: {source} (Score: {score:.3f})\n{node.text}\n"
            )
    
    context_text = "\n\n".join(context_parts) if context_parts else "No documents found."
    
    # Check if we have sufficient relevant content
    if not filtered_nodes:
        return {
            "answer": "I don't have sufficient relevant information in the knowledge base to answer this question.",
            "context": "No relevant documents found above the relevance threshold.",
            "retrieved_nodes": 0,
            "citations": [],
            "reranked": True,
            "reranker_type": "cohere" if _HAS_COHERE and os.getenv("COHERE_API_KEY") else "similarity",
            "confidence": "low"
        }
    
    # For specific terms, check if we have at least some relevant context
    # Removed overly strict 0.7 threshold that was blocking valid responses
    
    # Step 4: Generate contextual answer
    if context_parts:
        answer = f"""Based on the retrieved and re-ranked documents:

{context_text}

---
Retrieved {len(filtered_nodes)} contextually-enhanced chunks with citations above.
Use the [number] references to cite specific sources."""
    else:
        answer = "No relevant information found in the knowledge base."
    
    return {
        "answer": answer,
        "context": context_text,
        "retrieved_nodes": len(filtered_nodes),
        "citations": citations,
        "reranked": True,
        "reranker_type": "cohere" if _HAS_COHERE and os.getenv("COHERE_API_KEY") else "similarity",
        "confidence": "high" if len(filtered_nodes) >= 2 else "medium"
    }

@app.post("/ask-crewai")
def ask_with_crewai(body: AskRequest):
    """
    CrewAI-powered RAG endpoint with multi-agent orchestration.
    
    This endpoint uses a crew of three specialized agents working sequentially:
    1. Research Agent - Analyzes documents and extracts detailed information
    2. Synthesis Agent - Creates comprehensive, accurate answers
    3. Quality Agent - Validates and improves output quality
    
    This provides more thorough, validated answers compared to direct LLM calls,
    especially useful for complex or important questions requiring high accuracy.
    
    Pipeline:
    1. Retrieve initial candidates using vector similarity
    2. Re-rank using Cohere or similarity-based reranker
    3. Pass to CrewAI multi-agent system for processing
    4. Return quality-validated answer with citations
    """
    print(f"[CrewAI RAG] Received question: {body.question}")
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    
    # Check if LLM is available for CrewAI
    if llm is None:
        raise HTTPException(503, "LLM not available. CrewAI requires an active LLM connection.")
    
    # Step 1: Retrieve initial candidates (over-fetch for re-ranking)
    initial_top_k = (body.top_k or 5) * 2
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=initial_top_k,
    )
    
    nodes = retriever.retrieve(question)
    print(f"[CrewAI RAG] Retrieved {len(nodes)} initial candidates")
    
    # Step 2: Re-rank the nodes
    try:
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_str=question,
        )
        print(f"[CrewAI RAG] Re-ranked to {len(reranked_nodes)} nodes")
    except Exception as e:
        print(f"[CrewAI RAG] Re-ranking failed: {e}, using original nodes")
        reranked_nodes = nodes[:body.top_k or 5]
    
    # Step 3: Format context for CrewAI agents
    context_parts = []
    citations = []
    
    for i, node in enumerate(reranked_nodes[:body.top_k or 5]):
        source = node.metadata.get("source", "unknown")
        file_path = node.metadata.get("file_path", "")
        score = node.score if hasattr(node, 'score') else 0.0
        
        # Create citation
        citation = {
            "index": i + 1,
            "source": source,
            "file_path": file_path,
            "score": float(score),
            "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
        }
        citations.append(citation)
        
        # Format context with source info
        context_parts.append(
            f"[Source {i+1}] {source} (Relevance: {score:.3f})\n{node.text}\n"
        )
    
    retrieved_context = "\n\n".join(context_parts)
    
    # Step 4: Run CrewAI multi-agent workflow
    try:
        print(f"[CrewAI RAG] Starting multi-agent workflow...")
        crewai_answer = run_crewai_rag(question, retrieved_context)
        
        # Add source citations at the end if not already present
        if "**Sources:**" not in crewai_answer:
            citations_text = "\n\n**Sources:**\n" + "\n".join([
                f"[{i+1}] {node.metadata.get('source', 'unknown')} (relevance: {node.score if hasattr(node, 'score') else 0.0:.2f})"
                for i, node in enumerate(reranked_nodes[:body.top_k or 5])
            ])
            final_answer = crewai_answer + citations_text
        else:
            final_answer = crewai_answer
        
        return {
            "answer": final_answer,
            "retrieved_nodes": len(reranked_nodes),
            "method": "crewai_multi_agent",
            "agents_used": ["research_specialist", "synthesis_expert", "quality_assurance"],
            "workflow": "sequential",
            "citations": citations,
            "reranked": True,
            "reranker_type": "cohere" if _HAS_COHERE and os.getenv("COHERE_API_KEY") else "similarity"
        }
        
    except Exception as e:
        print(f"[CrewAI RAG] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"CrewAI workflow failed: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint with CrewAI multi-agent system.
    
    This endpoint provides OpenWebUI with:
    - CrewAI Multi-Agent Orchestration (Research + Synthesis + Quality)
    - Conversation Memory (persistent across turns)
    - Contextual RAG with Re-ranking
    - Citation Handling
    - Fallback to direct LLM if CrewAI fails
    
    OpenWebUI calls this endpoint and gets full multi-agent RAG capabilities.
    """
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user message provided")
    question = user_msgs[-1]
    
    # Generate deterministic session_id based on conversation start
    # This ensures the same conversation in OpenWebUI maintains memory
    conversation_start = json.dumps([
        {"role": m.role, "content": m.content[:100]} 
        for m in req.messages[:2]  # First 2 messages define the conversation
    ], sort_keys=True)
    
    session_id = f"webui-{hashlib.md5(conversation_start.encode()).hexdigest()}"
    
    print(f"[OpenWebUI] Question: {question}")
    print(f"[OpenWebUI] Session: {session_id}, Messages in request: {len(req.messages)}")
    
    # Get or create memory for this session
    memory = get_or_create_memory(session_id)
    existing_messages = memory.get_all()
    
    # Sync OpenWebUI conversation with our memory
    # Only add new messages that aren't already in memory
    if len(existing_messages) < len(req.messages):
        # Add missing messages from OpenWebUI to our memory
        for msg in req.messages[len(existing_messages):]:
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            memory.put(LLMChatMessage(role=role, content=msg.content))
    
    # Build context-aware query using conversation history
    chat_history = memory.get_all()
    context_query = question
    
    if len(chat_history) > 2:  # Has previous conversation context
        # Use last 4 messages (excluding current) for context
        recent_msgs = [msg for msg in chat_history[:-1]][-4:]
        recent_context = "\n".join([
            f"{msg.role}: {msg.content[:200]}" 
            for msg in recent_msgs
        ])
        context_query = f"Previous conversation:\n{recent_context}\n\nCurrent question: {question}"
        print(f"[OpenWebUI] Using {len(recent_msgs)} previous messages for context")
    
    # Retrieve initial candidates (over-fetch for re-ranking)
    initial_top_k = 10
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=initial_top_k,
    )
    
    nodes = retriever.retrieve(context_query)
    print(f"[OpenWebUI] Retrieved {len(nodes)} initial candidates")
    
    # Re-rank the nodes
    try:
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_str=question,  # Use original question for re-ranking
        )
        print(f"[OpenWebUI] Re-ranked to {len(reranked_nodes)} nodes")
    except Exception as e:
        print(f"[OpenWebUI] Re-ranking failed: {e}, using original nodes")
        reranked_nodes = nodes[:5]
    
    # Format context with citations
    context_parts = []
    for i, node in enumerate(reranked_nodes[:5]):
        source = node.metadata.get("source", "unknown")
        score = node.score if hasattr(node, 'score') else 0.0
        context_parts.append(
            f"**[{i+1}]** {source} (relevance: {score:.2f})\n{node.text}\n"
        )
    
    # Generate conversational answer with CrewAI multi-agent system
    if context_parts:
        context_text = "\n\n".join(context_parts)
        
        # Format context for CrewAI agents with conversation history
        crewai_context_parts = []
        for i, node in enumerate(reranked_nodes[:5]):
            source = node.metadata.get("source", "unknown")
            score = node.score if hasattr(node, 'score') else 0.0
            crewai_context_parts.append(
                f"[Source {i+1}] {source} (Relevance: {score:.3f})\n{node.text}\n"
            )
        
        # Add conversation context if exists
        conversation_context = ""
        if len(chat_history) > 2:
            recent_msgs = [msg for msg in chat_history[:-1]][-4:]
            conversation_context = "\n\n**Previous Conversation Context:**\n" + "\n".join([
                f"{msg.role}: {msg.content[:200]}" 
                for msg in recent_msgs
            ]) + "\n\n"
        
        retrieved_context = conversation_context + "\n\n".join(crewai_context_parts)
        
        # Use CrewAI multi-agent workflow for high-quality answers
        try:
            print(f"[OpenWebUI CrewAI] Starting multi-agent workflow...")
            crewai_answer = run_crewai_rag(question, retrieved_context)
            
            # Add source citations if not already present
            if "**Sources:**" not in crewai_answer:
                citations_text = "\n\n**Sources:**\n" + "\n".join([
                    f"[{i+1}] {node.metadata.get('source', 'unknown')} (relevance: {node.score if hasattr(node, 'score') else 0.0:.2f})"
                    for i, node in enumerate(reranked_nodes[:5])
                ])
                answer = crewai_answer + citations_text
            else:
                answer = crewai_answer
            
            print(f"[OpenWebUI CrewAI] Workflow completed successfully")
            
        except Exception as e:
            print(f"[OpenWebUI CrewAI] Workflow failed: {e}, falling back to direct LLM")
            # Fallback to direct LLM if CrewAI fails
            try:
                from llama_index.core.llms import ChatMessage as LLMChatMsg
                
                simple_context = "\n\n".join([
                    f"Source {i+1}: {node.text[:1000]}" 
                    for i, node in enumerate(reranked_nodes[:5])
                ])
                
                prompt = f"""{conversation_context}Context from documents:
{simple_context}

Question: {question}

Please provide a clear, accurate answer based ONLY on the information in the context above."""
                
                response = llm.chat([
                    LLMChatMsg(role="system", content="""You are a precise document analyst. RULES:
• Answer the question using ONLY the information explicitly present in the provided context
• Extract and present specific details like names, dates, terms, and definitions
• If the EXACT terms aren't found but RELATED information exists, explain what IS available
• Only say "I don't have sufficient information" if NO related information is found
• Be helpful - if you find relevant information, share it even if it uses different terminology
• Include full names with titles, organization names, dates, numbers, and roles when present
• Quote directly from source when possible
• NEVER make up information not in the context"""),
                    LLMChatMsg(role="user", content=prompt)
                ])
                
                llm_answer = str(response.message.content)
                
                # Validate response - only block if clearly stating no information at the start
                if llm_answer.strip().startswith("I don't have sufficient information"):
                    # If LLM clearly says no information, don't add sources
                    answer = "I don't have sufficient information in the provided documents to answer this question."
                else:
                    # Add source citations for all other responses
                    citations_text = "\n\n**Sources:**\n" + "\n".join([
                        f"[{i+1}] {node.metadata.get('source', 'unknown')} (relevance: {node.score if hasattr(node, 'score') else 0.0:.2f})"
                        for i, node in enumerate(reranked_nodes[:5])
                    ])
                    answer = llm_answer + citations_text
                
            except Exception as e2:
                print(f"[OpenWebUI] Fallback LLM also failed: {e2}, returning context")
                answer = f"""Based on the retrieved documents:

{context_text}

---
📚 Retrieved {len(reranked_nodes)} contextually-enhanced sources."""
    else:
        answer = "I don't have relevant information in my knowledge base to answer that question."
    
    # Update memory with assistant's response
    memory.put(LLMChatMessage(role=MessageRole.ASSISTANT, content=answer))
    print(f"[OpenWebUI] Memory updated. Total messages: {len(memory.get_all())}")
    
    # Return in OpenAI format for OpenWebUI compatibility
    return {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": req.model or OLLAMA_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(question.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(question.split()) + len(answer.split())
        },
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "llamaindex-rag", "object": "model", "owned_by": "local"},
        ]
    }

# ---- Chat History Management Endpoints ----

@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    """
    Get conversation history for a specific session.
    
    Returns all messages in the conversation with their roles and content.
    """
    if session_id not in chat_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found")
    
    memory = chat_sessions[session_id]
    messages = memory.get_all()
    
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {
                "role": str(msg.role),
                "content": msg.content,
            }
            for msg in messages
        ]
    }

@app.post("/chat/clear")
def clear_chat_history(body: ClearHistoryRequest):
    """
    Clear conversation history for a specific session.
    
    This removes all messages from the session's memory.
    """
    success = clear_session(body.session_id)
    
    if success:
        return {
            "status": "success",
            "message": f"Cleared history for session '{body.session_id}'",
            "session_id": body.session_id
        }
    else:
        raise HTTPException(404, f"Session '{body.session_id}' not found")

@app.get("/chat/sessions")
def list_sessions():
    """
    List all active chat sessions.
    
    Returns session IDs and message counts for all active conversations.
    """
    return {
        "active_sessions": len(chat_sessions),
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(memory.get_all()),
                "last_message": memory.get_all()[-1].content[:100] + "..." if memory.get_all() else None
            }
            for sid, memory in chat_sessions.items()
        ]
    }

@app.delete("/chat/sessions")
def clear_all_sessions():
    """
    Clear all chat sessions.
    
    WARNING: This removes all conversation history from all sessions.
    """
    session_count = len(chat_sessions)
    chat_sessions.clear()
    print(f"[Memory] Cleared all {session_count} sessions")
    
    return {
        "status": "success",
        "message": f"Cleared all {session_count} sessions",
        "sessions_cleared": session_count
    }

@app.get("/health")
def health():
    try:
        # Check vector store
        retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
        nodes = retriever.retrieve("test")
        return {
            "status": "ok",
            "vector_store": "LlamaIndex + PGVector",
            "nodes_available": len(nodes) > 0,
            "phoenix_enabled": _HAS_PHOENIX
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------- Phoenix Observability Endpoints ----------

@app.get("/phoenix/status")
def phoenix_status():
    """
    Get Phoenix observability status and configuration.
    """
    if not _HAS_PHOENIX:
        return {
            "enabled": False,
            "message": "Phoenix is not installed or failed to initialize",
            "install_command": "pip install arize-phoenix openinference-instrumentation-llama-index"
        }
    
    return {
        "enabled": True,
        "ui_url": "http://localhost:6006",
        "project_name": os.getenv("PHOENIX_PROJECT_NAME", "openwebui-rag"),
        "collector_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006"),
        "instrumented": {
            "llama_index": True,
            "crewai": False,  # Will be added next
        },
        "features": [
            "LLM call tracing",
            "Embedding tracking",
            "Retrieval monitoring",
            "Latency analysis",
            "Token usage tracking"
        ]
    }

@app.get("/phoenix/traces")
def get_recent_traces():
    """
    Get information about recent traces (if Phoenix is enabled).
    """
    if not _HAS_PHOENIX:
        raise HTTPException(404, "Phoenix observability is not enabled")
    
    return {
        "message": "View traces in Phoenix UI",
        "ui_url": "http://localhost:6006",
        "note": "Open the Phoenix UI to view detailed traces, spans, and metrics"
    }

@app.get("/phoenix/metrics")
def get_phoenix_metrics():
    """
    Get observability metrics summary.
    """
    if not _HAS_PHOENIX:
        raise HTTPException(404, "Phoenix observability is not enabled")
    
    return {
        "message": "Metrics available in Phoenix UI",
        "ui_url": "http://localhost:6006",
        "metrics_available": [
            "Total requests",
            "Average latency",
            "Token usage",
            "Error rates",
            "Retrieval quality",
            "LLM performance"
        ],
        "note": "Open Phoenix UI for detailed metrics and analytics"
    }

# ---------- RAGAs Evaluation Endpoints ----------

@app.get("/ragas/status")
def ragas_status():
    """
    Get RAGAs evaluation framework status.
    
    Returns configuration and available metrics.
    """
    if not _HAS_RAGAS:
        return {
            "enabled": False,
            "message": "RAGAs is not installed or failed to load",
            "install_command": "pip install ragas datasets",
            "documentation": "https://docs.ragas.io/"
        }
    
    return {
        "enabled": True,
        "metrics_available": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ],
        "metrics_description": {
            "faithfulness": "Measures if answer is grounded in given context (0-1)",
            "answer_relevancy": "Measures if answer addresses the question (0-1)",
            "context_precision": "Measures if retrieved contexts are relevant (0-1)",
            "context_recall": "Measures if all relevant info is retrieved (0-1, needs ground_truth)"
        },
        "llm_configured": ragas_llm is not None,
        "embeddings_configured": ragas_embeddings is not None,
        "status": "operational" if (ragas_llm and ragas_embeddings) else "partially_configured"
    }

@app.post("/evaluate")
def evaluate_single_query(
    question: str = Query(..., description="Question to evaluate"),
    ground_truth: str | None = Query(None, description="Reference answer for comparison"),
    top_k: int = Query(5, description="Number of contexts to retrieve")
):
    """
    Evaluate a single RAG query using RAGAs metrics.
    
    This endpoint:
    1. Retrieves contexts for the question
    2. Generates an answer using the RAG pipeline
    3. Evaluates using RAGAs metrics
    4. Returns scores and detailed results
    
    Metrics:
    - faithfulness: Answer grounded in context?
    - answer_relevancy: Answer addresses question?
    - context_precision: Retrieved docs are relevant?
    - context_recall: All relevant info retrieved? (needs ground_truth)
    """
    if not _HAS_RAGAS:
        raise HTTPException(503, "RAGAs evaluation framework is not available. Install with: pip install ragas datasets")
    
    if not (ragas_llm and ragas_embeddings):
        raise HTTPException(503, "RAGAs evaluators not properly configured")
    
    print(f"[RAGAs Evaluate] Question: {question}")
    
    # Step 1: Retrieve contexts
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k * 2,  # Over-fetch for re-ranking
    )
    
    nodes = retriever.retrieve(question)
    print(f"[RAGAs Evaluate] Retrieved {len(nodes)} candidates")
    
    # Step 2: Re-rank
    try:
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_str=question,
        )
        print(f"[RAGAs Evaluate] Re-ranked to {len(reranked_nodes)} nodes")
    except Exception as e:
        print(f"[RAGAs Evaluate] Re-ranking failed: {e}, using original nodes")
        reranked_nodes = nodes[:top_k]
    
    # Step 3: Extract contexts
    contexts = [node.text for node in reranked_nodes[:top_k]]
    
    # Step 4: Generate answer
    context_text = "\n\n".join([
        f"Context {i+1}: {node.text[:800]}" 
        for i, node in enumerate(reranked_nodes[:top_k])
    ])
    
    from llama_index.core.llms import ChatMessage as LLMChatMsg
    
    prompt = f"""Context from documents:
{context_text}

Question: {question}

Please provide a clear, accurate answer based ONLY on the information in the context above. Include specific details."""
    
    response = llm.chat([
        LLMChatMsg(role="system", content="You are a precise document analyst. Extract EXACT information from documents."),
        LLMChatMsg(role="user", content=prompt)
    ])
    
    answer = str(response.message.content)
    print(f"[RAGAs Evaluate] Generated answer: {answer[:100]}...")
    
    # Step 5: Evaluate with RAGAs
    print("[RAGAs Evaluate] Running evaluation metrics...")
    metrics = evaluate_rag_response(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth
    )
    
    print(f"[RAGAs Evaluate] Evaluation complete. Scores: {metrics}")
    
    # Step 6: Return results
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "num_contexts": len(contexts),
        "metrics": metrics,
        "sources": [
            {
                "index": i + 1,
                "source": node.metadata.get("source", "unknown"),
                "score": float(node.score if hasattr(node, 'score') else 0.0),
                "text_preview": node.text[:200] + "..."
            }
            for i, node in enumerate(reranked_nodes[:top_k])
        ],
        "evaluation_summary": {
            "total_metrics": len(metrics),
            "average_score": sum(v for k, v in metrics.items() if isinstance(v, (int, float))) / len([v for k, v in metrics.items() if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in metrics.values()) else 0.0,
            "has_errors": "error" in metrics
        }
    }

@app.post("/evaluate/batch")
def evaluate_batch(body: EvaluateRequest):
    """
    Evaluate multiple queries in batch using RAGAs.
    
    Useful for testing and benchmarking your RAG system.
    
    This endpoint processes multiple questions, generates answers,
    and evaluates them using RAGAs metrics. Returns individual
    scores for each question plus aggregate statistics.
    """
    if not _HAS_RAGAS:
        raise HTTPException(503, "RAGAs evaluation framework is not available")
    
    if not (ragas_llm and ragas_embeddings):
        raise HTTPException(503, "RAGAs evaluators not properly configured")
    
    questions = body.questions
    ground_truths = body.ground_truths or [None] * len(questions)
    top_k = body.top_k or 5
    
    if body.ground_truths and len(body.ground_truths) != len(questions):
        raise HTTPException(400, "Number of ground truths must match number of questions")
    
    print(f"[RAGAs Batch] Evaluating {len(questions)} questions")
    
    results = []
    all_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # Process each question
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        print(f"[RAGAs Batch] Processing {i+1}/{len(questions)}: {question}")
        
        try:
            # Retrieve and generate answer
            retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k * 2)
            nodes = retriever.retrieve(question)
            
            try:
                reranked_nodes = reranker.postprocess_nodes(nodes=nodes, query_str=question)
            except:
                reranked_nodes = nodes[:top_k]
            
            contexts = [node.text for node in reranked_nodes[:top_k]]
            
            # Generate answer
            context_text = "\n\n".join([
                f"Context {j+1}: {node.text[:800]}" 
                for j, node in enumerate(reranked_nodes[:top_k])
            ])
            
            from llama_index.core.llms import ChatMessage as LLMChatMsg
            prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer based on context:"
            response = llm.chat([
                LLMChatMsg(role="system", content="Extract information from context."),
                LLMChatMsg(role="user", content=prompt)
            ])
            answer = str(response.message.content)
            
            # Store data
            all_data["question"].append(question)
            all_data["answer"].append(answer)
            all_data["contexts"].append(contexts)
            all_data["ground_truth"].append(ground_truth or "")
            
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "num_contexts": len(contexts)
            })
            
        except Exception as e:
            print(f"[RAGAs Batch] Error processing question {i+1}: {e}")
            results.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": ground_truth,
                "error": str(e)
            })
    
    # Run batch evaluation
    print("[RAGAs Batch] Running batch evaluation...")
    dataset = Dataset.from_dict(all_data)
    
    metrics_list = [faithfulness, answer_relevancy, context_precision]
    if any(ground_truths):
        metrics_list.append(context_recall)
    
    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        
        # Add scores to results
        for i, result in enumerate(results):
            if "error" not in result:
                result["metrics"] = {}
                for metric_name, metric_values in eval_result.items():
                    if isinstance(metric_values, (list, tuple)) and i < len(metric_values):
                        result["metrics"][metric_name] = float(metric_values[i])
        
        # Calculate averages
        avg_metrics = {}
        for metric_name in eval_result.keys():
            values = eval_result[metric_name]
            if isinstance(values, (list, tuple)):
                valid_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))]  # Filter NaN
                if valid_values:
                    avg_metrics[f"avg_{metric_name}"] = float(sum(valid_values) / len(valid_values))
        
        evaluation_success = True
        
    except Exception as e:
        print(f"[RAGAs Batch] Batch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        avg_metrics = {"error": str(e)}
        evaluation_success = False
    
    return {
        "total_questions": len(questions),
        "successful_evaluations": len([r for r in results if "error" not in r]),
        "failed_evaluations": len([r for r in results if "error" in r]),
        "results": results,
        "average_metrics": avg_metrics,
        "evaluation_success": evaluation_success
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
