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

# CrewAI
from crewai import Agent, Task, Crew, Process

# Optional Docling support (PDF parsing)
try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except Exception as e:
    print(f"[Docling] Failed to import: {e}")
    DocumentConverter = None
    _HAS_DOCLING = False

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

# ---------- Ingestion Function ----------
def ingest_documents(folder: Path = DATA_DIR, patterns: tuple = ("*.txt", "*.md", "*.pdf")):
    """Ingest documents using LlamaIndex"""
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

    # Parse documents into nodes with chunking
    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    nodes = text_splitter.get_nodes_from_documents(documents)
    
    print(f"[LlamaIndex] Created {len(nodes)} nodes from {len(documents)} documents")
    
    # Insert nodes into vector store
    index.insert_nodes(nodes)
    
    print(f"[LlamaIndex] Ingested {len(nodes)} chunks")
    return len(nodes), [str(p) for p in files]

# ---------- CrewAI Agent ----------
llm_model = f"ollama/{OLLAMA_MODEL}"
os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE)

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
    llm=llm_model,
    verbose=False,
)

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

# ---- Endpoints ----
@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "data_dir": str(DATA_DIR),
        "vector_store": "LlamaIndex + PGVector",
    }

@app.post("/ingest")
def ingest(folder: str = Query(default=str(DATA_DIR), description="Folder to ingest")):
    n, files = ingest_documents(folder)
    return {
        "folder": str(Path(folder).resolve()),
        "files_seen": files,
        "ingested_chunks": n,
    }

@app.post("/ask")
def ask(body: AskRequest):
    print(f"[DEBUG] Received question: {body.question}")
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "question is required")

    # Use LlamaIndex retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=body.top_k or 5,
    )
    
    # Retrieve nodes
    nodes = retriever.retrieve(question)
    print(f"[DEBUG] Retrieved {len(nodes)} nodes")
    
    # Format context with citations
    context_parts = []
    for i, node in enumerate(nodes):
        source = node.metadata.get("source", "unknown")
        score = node.score if hasattr(node, 'score') else 0.0
        context_parts.append(
            f"[source: {source}, score: {score:.3f}]\n{node.text}"
        )
    
    context_text = "\n\n".join(context_parts) if context_parts else "No documents found."
    
    # Generate simple answer from retrieved context (without LLM)
    if context_parts:
        answer = f"Based on the retrieved documents:\n\n{context_text}"
    else:
        answer = "No relevant information found in the knowledge base."
    
    return {"answer": answer, "context": context_text, "retrieved_nodes": len(nodes)}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user message provided")
    question = user_msgs[-1]

    # Use LlamaIndex query engine
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
    
    # Query
    response = query_engine.query(question)
    text = str(response)

    return {
        "id": "chatcmpl-llamaindex-rag",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": req.model or OLLAMA_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "llamaindex-rag", "object": "model", "owned_by": "local"},
        ]
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
            "nodes_available": len(nodes) > 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
