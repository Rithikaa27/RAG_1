"""
Lightweight embedding engine using fastembed (ONNX-based).

Uses sentence-transformers/all-MiniLM-L6-v2 producing 384-dimensional vectors.
fastembed runs on ONNX Runtime — no PyTorch required, keeping bundle size
well under Vercel's 250 MB limit.

The model is downloaded once to cache_dir on first use (cold start ~5s),
then served from disk on subsequent warm invocations.
"""
import os
from typing import List

_embedder_instance = None


def _get_embedder():
    """Singleton: load the fastembed model once per process."""
    global _embedder_instance
    if _embedder_instance is None:
        from fastembed import TextEmbedding

        # On Vercel: /tmp is the only writable directory.
        # Locally: use a ./models/ folder (gitignored).
        cache_dir = os.environ.get(
            "FASTEMBED_CACHE_DIR",
            "/tmp/fastembed" if os.environ.get("VERCEL") else "./models/fastembed",
        )
        os.makedirs(cache_dir, exist_ok=True)

        _embedder_instance = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
        )
    return _embedder_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts.

    Returns list of 384-dimensional float vectors, one per input text.
    """
    if not texts:
        return []
    embedder = _get_embedder()
    return [emb.tolist() for emb in embedder.embed(texts)]


def embed_query(query: str) -> List[float]:
    """Embed a single query string. Returns a 384-dim float vector."""
    results = embed_texts([query])
    return results[0] if results else []
