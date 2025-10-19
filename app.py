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

import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from crewai import Agent, Task, Crew, Process  # CrewAI (no crewai_tools)

load_dotenv()

# ---------- Paths ----------
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"

# ---------- In-memory Vector Store ----------
class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # CPU for stability on macOS; change to device="cuda" if you have GPU
        self.embedder = SentenceTransformer(model_name, device="cpu")
        self.chunks: List[str] = []
        self.metadatas: List[Dict] = []
        self.embeddings: np.ndarray | None = None

    @staticmethod
    def _chunk(text: str, size: int = 500, overlap: int = 80) -> List[str]:
        tokens = text.split()
        out, step, i = [], max(1, size - overlap), 0
        while i < len(tokens):
            out.append(" ".join(tokens[i:i + size]))
            i += step
        return out

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)

    def ingest_folder(
        self,
        folder: str | Path = DATA_DIR,
        patterns: tuple[str, ...] = ("*.txt", "*.md"),
    ) -> tuple[int, list[str]]:
        folder = Path(folder).resolve()
        files: list[Path] = []
        for pat in patterns:
            files.extend(sorted(folder.rglob(pat)))

        texts, metas = [], []
        for p in files:
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"[ingest] skip {p}: {e}")
                continue
            if not raw:
                print(f"[ingest] skip empty {p}")
                continue
            for idx, chunk in enumerate(self._chunk(raw)):
                texts.append(chunk)
                metas.append({"source": p.name, "chunk": idx, "path": str(p)})

        if not texts:
            self.chunks, self.metadatas, self.embeddings = [], [], None
            return 0, [str(p) for p in files]

        self.chunks, self.metadatas = texts, metas
        self.embeddings = self._encode(texts)
        return len(texts), [str(p) for p in files]

    def query(self, q: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        qv = self._encode([q])[0]
        sims = self.embeddings @ qv  # cosine (embeddings are normalized)
        topk = np.argsort(-sims)[:k]
        return [(self.chunks[i], self.metadatas[i], float(sims[i])) for i in topk]


store = VectorStore()
store.ingest_folder(DATA_DIR)

# ---------- LiteLLM-style LLM selection for CrewAI ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

if OPENAI_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_KEY)
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # e.g., gpt-4o-mini
elif OLLAMA_BASE:
    llm_model = f"ollama/{OLLAMA_MODEL}"                   # e.g., ollama/llama3.2:3b
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

class AskRequest(BaseModel):
    question: str
    top_k: int | None = 5

@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "data_dir": str(DATA_DIR),
        "ingested_chunks": len(store.chunks),
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
    question = body.question.strip()
    k = body.top_k or 5

    results = store.query(question, k=k)
    if not results:
        context_text = "No local documents ingested. Use /ingest or add files to data/."
    else:
        parts = []
        for text, meta, score in results:
            src = f"{meta.get('source')}#chunk-{meta.get('chunk')}"
            parts.append(f"[source: {src}, score: {score:.3f}]\n{text}")
        context_text = "\n\n".join(parts)

    instructions = (
        "You are given a question and a set of retrieved context chunks.\n"
        "Write a concise answer grounded ONLY in those chunks. "
        "Cite sources in-line as [filename#chunk-N]. "
        "If the context is insufficient, say so.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n"
    )

    task = Task(description=instructions, agent=rag_agent, expected_output="A sourced answer.")
    crew = Crew(agents=[rag_agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

    return {"answer": str(result), "context": context_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
