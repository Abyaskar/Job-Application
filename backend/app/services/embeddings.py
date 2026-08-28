"""
Embedding provider abstraction.

Production systems generating candidate/job embeddings would call a hosted
embedding model (Vertex AI `text-embedding-004`, OpenAI `text-embedding-3`,
etc.). This project runs in a sandbox with no outbound access to those
APIs, so the default provider (`local_tfidf`) fits a TF-IDF vectorizer +
TruncatedSVD over the corpus to produce dense vectors locally, with zero
external dependencies. Everything downstream (vector index, ranking, RAG
retrieval) only depends on the `EmbeddingProvider` interface, so switching
EMBEDDING_PROVIDER=vertex_ai in production requires no changes outside this
file.

This is a deliberate, documented trade-off -- see README "Trade-offs".
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("services.embeddings")


class EmbeddingProvider(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalTfidfEmbeddingProvider(EmbeddingProvider):
    """TF-IDF + truncated SVD ("latent semantic") embeddings.

    Not a substitute for a transformer embedding model's semantic depth,
    but it is a real dense vector space with genuine cosine-similarity
    structure over vocabulary co-occurrence, which is enough to
    demonstrate (and evaluate) the retrieval/ranking architecture without
    external model downloads. `fit` must be called once per corpus
    (re-fit is triggered on cold start and can be scheduled periodically
    in production as the job corpus grows).
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            return
        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        n_components = min(self.dim, max(2, tfidf_matrix.shape[1] - 1), tfidf_matrix.shape[0] - 1)
        n_components = max(n_components, 2)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._fitted = True
        logger.info("embeddings.fitted", corpus_size=len(corpus), dims=n_components)

    def _fallback_hash_embedding(self, text: str) -> list[float]:
        """Deterministic fallback used before the corpus has been fit
        (e.g. embedding a single new resume at cold start)."""
        vec = np.zeros(self.dim)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed(self, text: str) -> list[float]:
        if not self._fitted:
            return self._fallback_hash_embedding(text)
        tfidf_vec = self._vectorizer.transform([text])
        dense = self._svd.transform(tfidf_vec)[0]
        norm = np.linalg.norm(dense)
        return (dense / norm if norm else dense).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class VertexAIEmbeddingProvider(EmbeddingProvider):
    """Production stub: Vertex AI `text-embedding-004` via google-cloud-aiplatform.

    Left unimplemented (raises) in this sandbox since it requires GCP
    credentials and network egress this environment does not have. Swap
    EMBEDDING_PROVIDER=vertex_ai and implement using
    `vertexai.language_models.TextEmbeddingModel` in a real GCP deployment.
    """

    def fit(self, corpus: list[str]) -> None:
        return  # hosted models require no local fitting

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Configure GCP credentials and implement Vertex AI embedding call here."
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Configure GCP credentials and implement Vertex AI embedding call here."
        )


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        settings = get_settings()
        if settings.EMBEDDING_PROVIDER == "vertex_ai":
            _provider = VertexAIEmbeddingProvider()
        else:
            _provider = LocalTfidfEmbeddingProvider(dim=settings.EMBEDDING_DIM)
    return _provider


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    if not va.any() or not vb.any():
        return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
