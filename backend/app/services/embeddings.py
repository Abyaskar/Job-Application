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

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name/version of the embedding model."""
        ...

    @property
    @abstractmethod
    def embedding_version(self) -> str:
        """Returns a version identifier for the embedding space.
        
        This is used to detect when old embeddings are incompatible
        with the current provider (e.g., TF-IDF vs transformer).
        """
        ...

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Returns the dimension of the embedding vectors."""
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

    @property
    def model_name(self) -> str:
        return "sklearn-tfidf-svd"

    @property
    def embedding_version(self) -> str:
        return f"tfidf-svd-v1-d{self.dim}"

    @property
    def embedding_dimension(self) -> int:
        return self.dim

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


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence-transformers embedding provider using pretrained models.

    This provider uses the sentence-transformers library to generate
    contextual embeddings from transformer models. The default model
    is 'sentence-transformers/all-MiniLM-L6-v2', which produces 384-dimensional
    embeddings with strong semantic understanding.

    Key characteristics:
    - Model is loaded once and reused (lazy initialization)
    - No corpus fitting required (pretrained model)
    - Produces normalized vectors suitable for cosine similarity
    - Runs entirely locally with no API calls
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dim = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_version(self) -> str:
        return f"st-{self._model_name.replace('/', '-')}-v1"

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def fit(self, corpus: list[str]) -> None:
        """No-op for pretrained models - they don't require fitting.
        
        Args:
            corpus: Ignored (pretrained model doesn't need fitting)
        """
        logger.info("embeddings.fit_skipped", reason="pretrained_model", model=self._model_name)
        return

    def embed(self, text: str) -> list[float]:
        """Generate a single embedding vector.
        
        Args:
            text: Text to embed
            
        Returns:
            L2-normalized embedding vector as list of floats
        """
        model = self._get_model()
        # Encode returns numpy array; normalize=True gives unit vectors
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of L2-normalized embedding vectors
        """
        if not texts:
            return []
        model = self._get_model()
        # Batch encoding is more efficient than individual calls
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


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
        elif settings.EMBEDDING_PROVIDER == "sentence_transformers":
            _provider = SentenceTransformerEmbeddingProvider()
        else:
            _provider = LocalTfidfEmbeddingProvider(dim=settings.EMBEDDING_DIM)
    return _provider


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text string.
    
    Uses the configured embedding provider (local TF-IDF by default,
    or external API in production).
    
    Args:
        text: The text to embed
        
    Returns:
        Dense vector as list of floats
    """
    provider = get_embedding_provider()
    return provider.embed(text)


def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts efficiently.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of dense vectors
    """
    provider = get_embedding_provider()
    return provider.embed_batch(texts)


def fit_embeddings(corpus: list[str]) -> None:
    """Fit the embedding model on a corpus.
    
    For local TF-IDF embeddings, this fits the vectorizer and SVD.
    For external APIs, this is a no-op.
    
    Args:
        corpus: List of documents to fit on
    """
    provider = get_embedding_provider()
    provider.fit(corpus)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    if not va.any() or not vb.any():
        return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
