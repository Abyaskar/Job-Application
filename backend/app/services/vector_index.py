"""
Vector index for candidate-to-job semantic retrieval.

At this project's scale (hundreds to low-thousands of jobs per candidate
search space) a brute-force cosine similarity scan held in memory is both
simpler and faster than standing up a dedicated ANN index -- O(n*d) over a
few thousand jobs is sub-millisecond. The `VectorIndex` interface is what
matters for portability: at production scale (hundreds of thousands+ of
jobs) this is the seam where you'd swap in MongoDB Atlas Vector Search
(if jobs already live in Mongo) or Vertex AI Vector Search / a dedicated
ANN store (FAISS, ScaNN) behind the same `search()` signature, with no
change to the ranking or RAG layers that consume it.
"""
from __future__ import annotations

import numpy as np

from app.services.embeddings import cosine_similarity


class VectorIndex:
    def __init__(self):
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._matrix: np.ndarray | None = None

    def build(self, items: list[tuple[str, list[float]]]) -> None:
        self._ids = [i for i, _ in items]
        self._vectors = [v for _, v in items]
        self._matrix = np.array(self._vectors) if self._vectors else None

    def upsert(self, item_id: str, vector: list[float]) -> None:
        if item_id in self._ids:
            idx = self._ids.index(item_id)
            self._vectors[idx] = vector
        else:
            self._ids.append(item_id)
            self._vectors.append(vector)
        self._matrix = np.array(self._vectors) if self._vectors else None

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = np.array(query_vector)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        mat_norms = np.linalg.norm(self._matrix, axis=1)
        mat_norms[mat_norms == 0] = 1e-9
        sims = (self._matrix @ q) / (mat_norms * q_norm)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._ids)


_job_index = VectorIndex()


def get_job_vector_index() -> VectorIndex:
    return _job_index
