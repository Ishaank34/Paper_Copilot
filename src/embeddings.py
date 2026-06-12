"""
Embedding Module for Research Papers

Generates high-quality semantic embeddings for research paper chunks using Sentence Transformers.
Optimized for computer science and general research papers.
Uses BAAI/bge-small-en-v1.5 model for efficient semantic representation.

Features:
- Model caching for single-load efficiency
- Batch processing for scalable embedding generation
- NumPy array outputs for compatibility
- Comprehensive error handling and logging
- Cosine similarity computation
"""

import logging
from typing import List, Union, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)

# Global model cache
_embedding_model: Optional[SentenceTransformer] = None
_model_name: str = "BAAI/bge-small-en-v1.5"

# Model configuration
EMBEDDING_DIMENSION = 384  # BAAI/bge-small-en-v1.5 output dimension
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 256  # Memory-safe upper bound
MIN_BATCH_SIZE = 1


def load_embedding_model(model_name: str = _model_name) -> SentenceTransformer:
    """
    Load and cache the embedding model.

    Uses a global cache to ensure the model is loaded only once per session,
    avoiding redundant memory allocation and initialization overhead.

    Args:
        model_name: Name of the model to load (default: BAAI/bge-small-en-v1.5).
                    Can be changed for alternative models.

    Returns:
        Loaded SentenceTransformer model instance.

    Raises:
        Exception: If model loading fails (network issues, invalid model name, etc.).

    Example:
        >>> model = load_embedding_model()
        >>> embeddings = model.encode("Your text here")
    """
    global _embedding_model

    if _embedding_model is not None:
        logger.debug(f"Using cached embedding model: {model_name}")
        return _embedding_model

    try:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        logger.info(
            f"Successfully loaded model {model_name} "
            f"(Output dimension: {EMBEDDING_DIMENSION})"
        )
        return _embedding_model

    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.

    Embeds a single text string using the cached model. Useful for
    embedding individual queries or small text samples.

    Args:
        text: Text string to embed. Should be meaningful content
              (typically 50+ words for best results).

    Returns:
        NumPy array of shape (384,) containing the embedding vector.

    Raises:
        ValueError: If text is empty or None.
        Exception: If model loading or encoding fails.

    Example:
        >>> embedding = embed_text("Introduction to transformers in NLP")
        >>> print(embedding.shape)
        (384,)
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    try:
        model = load_embedding_model()
        # Encode with normalize_embeddings=True for cosine similarity compatibility
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        logger.debug(f"Embedded text of length {len(text)} characters")
        return embedding

    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.

    Embeds multiple text chunks efficiently using the cached model.
    Internally uses batch processing for optimal memory usage.

    Args:
        chunks: List of text strings to embed. Each chunk should be
                meaningful content (typically 300-1000 words for paper sections).

    Returns:
        NumPy array of shape (num_chunks, 384) containing embedding vectors.
        Rows correspond to input chunks in order.

    Raises:
        ValueError: If chunks is empty or contains invalid data.
        Exception: If model loading or encoding fails.

    Example:
        >>> chunks = ["Section 1 content...", "Section 2 content..."]
        >>> embeddings = embed_chunks(chunks)
        >>> print(embeddings.shape)
        (2, 384)
    """
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Chunks must be a non-empty list")

    if not all(isinstance(c, str) for c in chunks):
        raise ValueError("All chunks must be strings")

    if len(chunks) == 0:
        raise ValueError("Chunks list cannot be empty")

    try:
        logger.info(f"Embedding {len(chunks)} chunks")
        embeddings = batch_embed(chunks, batch_size=DEFAULT_BATCH_SIZE)
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return embeddings

    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        raise


def batch_embed(
    chunks: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """
    Generate embeddings for chunks using batch processing.

    Processes chunks in batches to optimize memory usage and computation.
    Batch processing is significantly faster than sequential embedding
    for multiple chunks.

    Args:
        chunks: List of text strings to embed.
        batch_size: Number of chunks to process per batch.
                   Default: 32. Valid range: 1-256.
                   Larger batches are faster but use more memory.
                   Typical chunk size: 300-1000 words (~50-200 tokens).

    Returns:
        NumPy array of shape (num_chunks, 384) containing embedding vectors.

    Raises:
        ValueError: If chunks is empty, batch_size is invalid, or chunks contain invalid data.
        Exception: If model loading or encoding fails.

    Notes:
        - Batch size of 32 balances speed and memory usage
        - Typical encoding time: 10-20ms per batch on CPU, faster on GPU
        - Memory overhead: approximately 1.5MB per batch (for batch_size=32)

    Example:
        >>> chunks = ["Text chunk 1", "Text chunk 2", "Text chunk 3"]
        >>> embeddings = batch_embed(chunks, batch_size=32)
        >>> print(embeddings.shape)
        (3, 384)
    """
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Chunks must be a non-empty list")

    if not all(isinstance(c, str) for c in chunks):
        raise ValueError("All chunks must be strings")

    if len(chunks) == 0:
        raise ValueError("Chunks list cannot be empty")

    # Validate and clamp batch size
    if not isinstance(batch_size, int) or batch_size < MIN_BATCH_SIZE:
        logger.warning(
            f"Invalid batch size {batch_size}, using minimum {MIN_BATCH_SIZE}"
        )
        batch_size = MIN_BATCH_SIZE

    if batch_size > MAX_BATCH_SIZE:
        logger.warning(
            f"Batch size {batch_size} exceeds maximum {MAX_BATCH_SIZE}, "
            f"clamping to {MAX_BATCH_SIZE}"
        )
        batch_size = MAX_BATCH_SIZE

    try:
        model = load_embedding_model()
        logger.debug(
            f"Batch embedding {len(chunks)} chunks with batch_size={batch_size}"
        )

        # Encode all chunks at once with batch processing
        embeddings = model.encode(
            chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=True
        )

        logger.debug(
            f"Generated embeddings with shape {embeddings.shape} "
            f"({len(chunks)} chunks × {EMBEDDING_DIMENSION} dimensions)"
        )
        return embeddings

    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        raise


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Computes the cosine similarity score between two embedding vectors.
    Ranges from -1 (opposite) to 1 (identical). With normalized embeddings,
    ranges from 0 to 1 (identical to opposite).

    Args:
        v1: First embedding vector of shape (384,) or (1, 384).
        v2: Second embedding vector of shape (384,) or (1, 384).

    Returns:
        Cosine similarity score as a float between -1 and 1.
        Higher values indicate greater semantic similarity.

    Raises:
        ValueError: If vectors have incompatible shapes or dimensions.
        TypeError: If vectors are not numpy arrays or convertible to arrays.

    Formula:
        cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)
        With normalized vectors: cosine_similarity = v1 · v2

    Example:
        >>> v1 = np.array([0.1, 0.2, 0.3])
        >>> v2 = np.array([0.1, 0.2, 0.3])
        >>> similarity = cosine_similarity(v1, v2)
        >>> print(f"Similarity: {similarity:.4f}")  # Output: ~1.0
    """
    try:
        # Convert to numpy arrays if needed
        if not isinstance(v1, np.ndarray):
            v1 = np.array(v1)
        if not isinstance(v2, np.ndarray):
            v2 = np.array(v2)

        # Flatten arrays if needed (handle both (384,) and (1, 384) shapes)
        v1 = v1.flatten()
        v2 = v2.flatten()

        # Validate dimensions
        if v1.shape != v2.shape:
            raise ValueError(
                f"Vector dimensions must match: {v1.shape} vs {v2.shape}"
            )

        if len(v1.shape) != 1:
            raise ValueError(
                f"Vectors must be 1-dimensional, got shape {v1.shape}"
            )

        # Check for zero vectors (edge case)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            logger.warning("One or both vectors have zero magnitude")
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(v1, v2) / (v1_norm * v2_norm)

        # Clamp to [-1, 1] to handle floating-point precision errors
        similarity = np.clip(similarity, -1.0, 1.0)

        logger.debug(f"Computed cosine similarity: {similarity:.4f}")
        return float(similarity)

    except ValueError as e:
        logger.error(f"ValueError in cosine similarity: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in cosine similarity: {e}")
        raise


def batch_cosine_similarity(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between a query and multiple chunks.

    Efficiently computes cosine similarity scores between a single query
    embedding and multiple chunk embeddings using vectorized operations.

    Args:
        query_embedding: Single embedding vector of shape (384,).
        chunk_embeddings: Multiple embedding vectors of shape (num_chunks, 384).

    Returns:
        NumPy array of similarity scores of shape (num_chunks,),
        with values between -1 and 1.

    Raises:
        ValueError: If embedding dimensions are incompatible.
        TypeError: If inputs are not numpy arrays.

    Example:
        >>> query = np.random.randn(384)
        >>> chunks = np.random.randn(10, 384)
        >>> similarities = batch_cosine_similarity(query, chunks)
        >>> print(similarities.shape)
        (10,)
        >>> best_match_idx = np.argmax(similarities)
    """
    try:
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(chunk_embeddings, np.ndarray):
            chunk_embeddings = np.array(chunk_embeddings)

        query_embedding = query_embedding.flatten()

        # Validate dimensions
        if query_embedding.shape[0] != chunk_embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} "
                f"must match chunk embeddings dimension {chunk_embeddings.shape[1]}"
            )

        # Vectorized cosine similarity computation
        # With normalized embeddings, this is simply matrix multiplication
        similarities = chunk_embeddings @ query_embedding

        # Clamp to [-1, 1]
        similarities = np.clip(similarities, -1.0, 1.0)

        logger.debug(
            f"Computed batch similarities for {len(similarities)} chunks"
        )
        return similarities

    except Exception as e:
        logger.error(f"Error in batch cosine similarity: {e}")
        raise


# Utility function for model info
def get_model_info() -> dict:
    """
    Get information about the loaded embedding model.

    Returns:
        Dictionary containing model metadata and configuration.

    Example:
        >>> info = get_model_info()
        >>> print(info['embedding_dimension'])
        384
    """
    try:
        model = load_embedding_model()
        return {
            "model_name": _model_name,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_batch_size": MAX_BATCH_SIZE,
            "model_loaded": _embedding_model is not None,
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise
