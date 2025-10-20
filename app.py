from __future__ import annotations

# ---- macOS / TF guards (keep immediately after the future import) ----
import os as _os

_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# ----------------------------------------------------------------------

import os, glob
from pathlib import Path
from typing import List, Tuple, Dict
import json

import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException
import asyncio
from crewai import Agent, Task, Crew, Process  # CrewAI (no crewai_tools)
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# Optional Docling support (PDF parsing). If unavailable, we skip PDFs gracefully.
try:
    from docling.document_converter import DocumentConverter  # type: ignore
    _HAS_DOCLING = True
except Exception as e:
    print(f"[Docling] Failed to import: {e}")
    DocumentConverter = None  # type: ignore
    _HAS_DOCLING = False

load_dotenv()

# ---------- Paths ----------
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"

# ---------- PostgreSQL Vector Store ----------

from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

# NEW: single global Docling converter (lazy init)
_DOC_CONVERTER: DocumentConverter | None = None


def _get_docling() -> DocumentConverter | None:
    if not _HAS_DOCLING:
        print("[Docling] Docling package not available")
        return None
    global _DOC_CONVERTER
    if _DOC_CONVERTER is None:
        try:
            _DOC_CONVERTER = DocumentConverter()  # type: ignore[call-arg]
            print("[Docling] DocumentConverter initialized successfully")
        except Exception as e:
            print(f"[Docling] Failed to initialize DocumentConverter: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _DOC_CONVERTER


# --- replace _read_text_from_path with this ---
def _read_text_from_path(p: Path) -> str:
    """
    Returns text for .txt/.md; uses Docling for .pdf.
    Logs parse stats so you can see why a file yields 0 chunks.
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
            # Try to convert the PDF
            result = None
            try:
                # Try the simplest API first (path as string)
                result = converter.convert(str(p))
            except Exception as e1:
                print(f"[Docling] Direct convert failed for {p}: {e1}")
                return ""

            # Extract document from result
            doc = getattr(result, "document", None) if result else None
            if doc is None:
                print(f"[Docling] No document in result for {p}")
                return ""

            # Try to export as markdown first (preferred)
            try:
                if hasattr(doc, "export_to_markdown"):
                    md = doc.export_to_markdown()
                    if md and md.strip():
                        print(f"[Docling] Parsed {p.name}: markdown length={len(md)}")
                        return md
            except Exception as e:
                print(f"[Docling] export_to_markdown failed: {e}")

            # Fallback to text export
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

    # other extensions -> ignored
    return ""


class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", database_url: str = None):
        self.embedder = SentenceTransformer(model_name, device="cpu")
        # IMPORTANT: default to the Docker service, not localhost
        self.database_url = database_url or os.getenv(
            "DATABASE_URL"
        )
        self.engine = create_engine(self.database_url, future=True)
        self._init_database()

    def _init_database(self):
        with self.engine.begin() as conn:
            # enable extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # create table (embedding dimension = 384 for MiniLM-L6-v2; set 768 if you change models)
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
              id SERIAL PRIMARY KEY,
              content TEXT NOT NULL,
              embedding VECTOR(384) NOT NULL,
              metadata JSONB,
              source TEXT,
              chunk_index INT
            )
            """))

    @staticmethod
    def _chunk(text: str, size: int = 500, overlap: int = 80):
        tokens = text.split()
        out, step, i = [], max(1, size - overlap), 0
        while i < len(tokens):
            out.append(" ".join(tokens[i:i + size]))
            i += step
        return out

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def ingest_folder(
            self,
            folder: str | Path = DATA_DIR,
            patterns: tuple[str, ...] = ("*.txt", "*.md", "*.pdf")  # NEW: include PDFs
    ) -> tuple[int, list[str]]:
        folder = Path(folder).resolve()
        files: list[Path] = []
        for pat in patterns:
            files.extend(sorted(folder.rglob(pat)))

        texts, metas = [], []
        for p in files:
            try:
                raw = _read_text_from_path(p)
                raw = (raw or "").strip()
            except Exception as e:
                print(f"[ingest] skip {p}: {e}")
                continue
            if not raw:
                continue
            for idx, chunk in enumerate(self._chunk(raw)):
                texts.append(chunk)
                metas.append({"source": p.name, "chunk": idx, "path": str(p)})

        if not texts:
            return 0, [str(p) for p in files]

        embeddings = self._encode(texts)

        with self.engine.begin() as conn:
            # clear any old rows from this path
            conn.execute(
                text("DELETE FROM documents WHERE metadata->>'path' LIKE :pattern"),
                {"pattern": f"{folder}%"}
            )
            # bulk insert; embed and metadata literals are inlined to avoid parameter cast issues
            for text_content, meta, emb in zip(texts, metas, embeddings):
                emb_lit = "[" + ",".join(str(float(x)) for x in emb.tolist()) + "]"
                meta_json = json.dumps(meta).replace("'", "''")
                sql = f"""
                      INSERT INTO documents (content, embedding, metadata, source, chunk_index)
                      VALUES (:content, '{emb_lit}'::vector, '{meta_json}'::jsonb, :source, :chunk_index)
                    """
                conn.execute(
                    text(sql),
                    {
                        "content": text_content,
                        "source": meta.get("source", ""),
                        "chunk_index": int(meta.get("chunk", 0)),
                    }
                )
        print(f"[VectorStore] Stored {len(texts)} documents")
        return len(texts), [str(p) for p in files]

    def query(self, q: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        q_emb = self._encode([q])[0]
        q_lit = "[" + ",".join(str(float(x)) for x in q_emb.tolist()) + "]"
        with self.engine.begin() as conn:
            sql = f"""
                  SELECT content, metadata, 1 - (embedding <=> '{q_lit}'::vector) AS similarity
                  FROM documents
                  ORDER BY embedding <=> '{q_lit}'::vector
                  LIMIT :k
                """
            rows = conn.execute(text(sql), {"k": k}).fetchall()
        out = []
        for content, metadata, sim in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            out.append((content, meta, float(sim)))
        return out

    def get_chunk_count(self) -> int:
        with self.engine.begin() as conn:
            return int(conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0)


store = VectorStore()
# Do not auto-ingest at startup; call /ingest explicitly after DB is ready

# ---------- LiteLLM-style LLM selection for CrewAI ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

if OPENAI_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_KEY)
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # e.g., gpt-4o-mini
elif OLLAMA_BASE:
    llm_model = f"ollama/{OLLAMA_MODEL}"  # e.g., ollama/llama3.2:3b
    os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE)  # e.g., http://localhost:11434
else:
    raise RuntimeError(
        "No LLM configured. Set OPENAI_API_KEY (and optional OPENAI_MODEL) "
        "or OLLAMA_BASE_URL and OLLAMA_MODEL."
    )

# ---------- CrewAI Agent (no external tools) ----------
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
    llm=llm_model,  # <- pass model string (LiteLLM style)
    verbose=False,
)

# ---------- FastAPI ----------
app = FastAPI(title="Simple RAG (FastAPI + CrewAI)")

# add near the top with other imports
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from datetime import datetime

# after `app = FastAPI(...)` add CORS (OpenWebUI runs on :3000 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Minimal OpenAI-compatible /v1/chat/completions ----
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool | None = False
    temperature: float | None = 0.2


import httpx


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user message provided")
    question = user_msgs[-1]

    # retrieve top-k
    results = store.query(question, k=5)
    context = "No local documents ingested." if not results else "\n\n".join(
        f"[source: {m.get('source')}#chunk-{m.get('chunk')}, score: {s:.3f}]\n{t}"
        for t, m, s in results
    )
    prompt = (
        "Answer ONLY from the context, cite like [file#chunk-N]. If insufficient, say so.\n\n"
        f"Question:\n{question}\n\nContext:\n{context}\n"
    )

    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": req.temperature or 0.2}
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "")

    from datetime import datetime
    return {
        "id": "chatcmpl-ollama-rag",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": payload["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class AskRequest(BaseModel):
    question: str
    top_k: int | None = 5


@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "data_dir": str(DATA_DIR),
        "ingested_chunks": store.get_chunk_count(),
    }


@app.post("/ingest")
def ingest(folder: str = Query(default=str(DATA_DIR), description="Folder to ingest")):
    n, files = store.ingest_folder(folder)
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

    print(f"[DEBUG] Starting vector query...")
    # retrieve
    results = store.query(question, k=body.top_k or 5)
    print(f"[DEBUG] Vector query completed, found {len(results)} results")
    context_text = "No local documents ingested. Use /ingest or add files to data/."
    if results:
        context_text = "\n\n".join(
            f"[source: {m.get('source')}#chunk-{m.get('chunk')}, score: {score:.3f}]\n{text}"
            for text, m, score in results
        )

    # For now, return a simple response to test the vector query
    # TODO: Re-enable CrewAI once Ollama connection is fixed
    answer = f"Based on the context: {context_text[:200]}..."
    print(f"[DEBUG] Returning simple answer: {answer[:100]}...")

    return {"answer": answer, "context": context_text}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "crew-rag222", "object": "model", "owned_by": "local"},
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "ingested_chunks": store.get_chunk_count()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
