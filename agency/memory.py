"""Chroma vector memory for the main agent.

Stores past chats, tasks, and worker outputs in ./chroma_db so the
boss can recall relevant context when routing (chat vs task).

Default embedding is a tiny hash-based function (no downloads, works
offline). Set MEMORY_EMBEDDING=default to use Chroma's ONNX MiniLM.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
import uuid

import config


class HashEmbedding:
    """Deterministic offline embedding: hashed bag-of-tokens, L2-normalized."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:  # required by newer chroma validation
        return "hash-embed"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - chroma API
        return [self._embed(t) for t in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - chroma 1.x API
        return self.__call__(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - chroma 1.x API
        return self.__call__(input)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _embedding_fn():
    if os.getenv("MEMORY_EMBEDDING", "hash") == "default":
        return None  # chroma default (ONNX, downloads on first use)
    return HashEmbedding()


class AgencyMemory:
    def __init__(self, path: str | None = None, collection: str = "agency_memory",
                 top_k: int | None = None, enabled: bool | None = None):
        self.path = path or config.CHROMA_PATH
        self.collection_name = collection
        self.top_k = top_k or config.MEMORY_TOP_K
        self.enabled = config.MEMORY_ENABLED if enabled is None else enabled
        self._client = None
        self._col = None

    def _connect(self):
        if self._col is not None:
            return self._col
        import chromadb

        if self.path == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=self.path)
        ef = _embedding_fn()
        if ef is None:
            self._col = self._client.get_or_create_collection(self.collection_name)
        else:
            self._col = self._client.get_or_create_collection(
                self.collection_name, embedding_function=ef
            )
        return self._col

    # -- writes ---------------------------------------------------------
    def add(self, text: str, kind: str = "exchange", **meta) -> str | None:
        if not self.enabled or not text or not text.strip():
            return None
        col = self._connect()
        doc_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        col.add(ids=[doc_id], documents=[text[:8000]],
                metadatas=[{"kind": kind, "ts": time.time(), **meta}])
        return doc_id

    def remember_exchange(self, user_msg: str, assistant_msg: str, mode: str = "chat") -> str | None:
        return self.add(f"User: {user_msg}\nAssistant: {assistant_msg}", kind="exchange", mode=mode)

    def remember_task(self, goal: str, tasks: list[dict], final: str) -> str | None:
        summary = "; ".join(f"{t['agent']}:{t['instruction'][:80]}" for t in tasks)
        return self.add(f"Task: {goal}\nPlan: {summary}\nResult: {final[:2000]}",
                        kind="task", mode="task")

    # -- reads ----------------------------------------------------------
    def recall(self, query: str, n: int | None = None) -> list[str]:
        if not self.enabled or not query or not query.strip():
            return []
        col = self._connect()
        if col.count() == 0:
            return []
        res = col.query(query_texts=[query], n_results=n or self.top_k)
        docs = (res.get("documents") or [[]])[0]
        return [d for d in docs if d]

    def recall_context(self, query: str, max_chars: int | None = None) -> str:
        docs = self.recall(query)
        if not docs:
            return ""
        ctx = "\n---\n".join(d[:1200] for d in docs)
        return ctx[: max_chars or config.MEMORY_MAX_CHARS]

    def count(self) -> int:
        if not self.enabled:
            return 0
        return self._connect().count()

    def clear(self) -> None:
        if not self.enabled:
            return
        self._client.delete_collection(self.collection_name)
        self._col = None
