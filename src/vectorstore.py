"""
Vector Store Module for Research Paper Embeddings

Manages efficient storage and retrieval of paper chunk embeddings using FAISS.
Supports semantic search with cosine similarity across multiple papers.
Optimized for retrieval quality with support for future scalability upgrades.

Features:
- FAISS-based indexing for fast similarity search
- Metadata persistence alongside embeddings
- Multi-paper support with tracking
- Cosine similarity search (normalized embeddings)
- Top-k retrieval with ranking
- Comprehensive error handling and logging
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import faiss

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_INDEXES_FOLDER = "indexes"
INDEX_FILE_NAME = "faiss_index.bin"
METADATA_FILE_NAME = "metadata.json"


class VectorStore:
    """
    Vector store for efficiently storing and retrieving paper chunk embeddings.

    Uses FAISS (Facebook AI Similarity Search) for fast nearest neighbor search
    with cosine similarity. Supports metadata tracking for multi-paper retrieval.

    Attributes:
        dimension: Embedding dimension (384 for BAAI/bge-small-en-v1.5)
        index: FAISS index object for similarity search
        metadata: List of metadata dictionaries for each embedding
        index_path: Path where index and metadata are saved
    """

    def __init__(
        self,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        index_folder: str = DEFAULT_INDEXES_FOLDER,
    ):
        """
        Initialize a new vector store.

        Args:
            dimension: Embedding dimension. Must be 384 for BAAI/bge-small-en-v1.5.
            index_folder: Folder path for saving/loading indexes (default: "indexes").

        Raises:
            ValueError: If dimension is invalid or not 384.

        Example:
            >>> vs = VectorStore(dimension=384)
            >>> print(f"Vector store initialized: {vs.dimension}D")
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")

        if dimension != DEFAULT_EMBEDDING_DIMENSION:
            logger.warning(
                f"Non-standard dimension {dimension}. "
                f"Recommended: {DEFAULT_EMBEDDING_DIMENSION}"
            )

        self.dimension = dimension
        self.index_folder = Path(index_folder)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        # Create index folder if it doesn't exist
        self.index_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Vector store initialized: {dimension}D, folder: {index_folder}")

    def create_index(self) -> None:
        """
        Create an empty FAISS index for similarity search.

        Creates an IndexFlatIP (Inner Product) index which, with normalized
        embeddings, provides cosine similarity search. This is the optimal
        choice for retrieval quality on academic papers.

        Raises:
            Exception: If index creation fails.

        Example:
            >>> vs = VectorStore()
            >>> vs.create_index()
            >>> print(f"Index created, size: {vs.index.ntotal}")
            Index created, size: 0
        """
        try:
            # IndexFlatIP: Inner Product search
            # With normalized embeddings, this equals cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

            logger.info(
                f"Created IndexFlatIP with dimension {self.dimension} "
                f"(supports up to {2**63-1} vectors)"
            )

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """
        Add embeddings to the vector store with associated metadata.

        Adds embeddings to the FAISS index and tracks metadata for retrieval.
        Embeddings should be normalized (L2 norm = 1) for cosine similarity.

        Args:
            embeddings: NumPy array of shape (num_embeddings, 384).
                       Should be normalized for cosine similarity.
            metadata_list: List of metadata dictionaries, one per embedding.
                          Each dict should contain: paper_name, section_name,
                          chunk_text, page_number, start_idx, end_idx.

        Raises:
            ValueError: If embeddings/metadata dimensions don't match or are invalid.
            RuntimeError: If index hasn't been created.

        Notes:
            - Embeddings must be float32 numpy arrays
            - Should be L2-normalized for accurate cosine similarity
            - Metadata list length must match embeddings count

        Example:
            >>> vs = VectorStore()
            >>> vs.create_index()
            >>> embeddings = np.random.randn(5, 384).astype(np.float32)
            >>> embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            >>> metadata = [{"paper": f"paper_{i}", "section": "intro"} for i in range(5)]
            >>> vs.add_embeddings(embeddings, metadata)
            >>> print(f"Index now contains {vs.index.ntotal} vectors")
        """
        if self.index is None:
            raise RuntimeError(
                "Index not created. Call create_index() first."
            )

        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a NumPy array")

        if embeddings.dtype != np.float32:
            logger.warning(
                f"Converting embeddings from {embeddings.dtype} to float32"
            )
            embeddings = embeddings.astype(np.float32)

        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Embeddings must be 2D array (num_embeddings, dimension), "
                f"got shape {embeddings.shape}"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of metadata entries ({len(metadata_list)})"
            )

        if len(embeddings) == 0:
            logger.warning("Adding zero embeddings to index")
            return

        try:
            # Verify embeddings are normalized (for cosine similarity)
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5):
                logger.warning(
                    "Embeddings appear to not be L2-normalized. "
                    "Cosine similarity may be inaccurate."
                )

            # Add embeddings to FAISS index
            self.index.add(embeddings)
            self.metadata.extend(metadata_list)

            logger.info(
                f"Added {len(embeddings)} embeddings to index. "
                f"Index size: {self.index.ntotal} total"
            )

        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k most similar embeddings using cosine similarity.

        Returns the k nearest neighbors to the query embedding, ranked by
        similarity score (higher is more similar).

        Args:
            query_embedding: Query embedding of shape (384,).
                            Should be L2-normalized.
            k: Number of top results to return (default: 5).
               Must be <= index size.

        Returns:
            List of result dictionaries, sorted by similarity (descending):
            [{
                'metadata': {...},
                'score': 0.95,          # Cosine similarity [0, 1]
                'rank': 1
            }, ...]

        Raises:
            ValueError: If query embedding is invalid or k is out of range.
            RuntimeError: If index is empty or not created.

        Notes:
            - Query must be L2-normalized for accurate cosine similarity
            - Scores range from 0 to 1 (normalized dot product)
            - With normalized embeddings, score = cosine similarity

        Example:
            >>> vs = VectorStore()
            >>> vs.create_index()
            >>> # ... add embeddings ...
            >>> query = np.random.randn(384).astype(np.float32)
            >>> query = query / np.linalg.norm(query)
            >>> results = vs.search(query, k=3)
            >>> for rank, result in enumerate(results, 1):
            ...     print(f"{rank}. {result['metadata']['section']} - {result['score']:.3f}")
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError(
                "Index is empty. Create index and add embeddings first."
            )

        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be a NumPy array")

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        if len(query_embedding.shape) != 1:
            raise ValueError(
                f"Query embedding must be 1D array of shape ({self.dimension},), "
                f"got shape {query_embedding.shape}"
            )

        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[0]}"
            )

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        if k > self.index.ntotal:
            logger.warning(
                f"k ({k}) exceeds index size ({self.index.ntotal}). "
                f"Clamping to index size."
            )
            k = self.index.ntotal

        try:
            # Reshape query for FAISS (needs to be 2D)
            query_reshaped = query_embedding.reshape(1, -1)

            # Search: returns (distances, indices)
            # distances are inner products (= cosine sim for normalized embeddings)
            distances, indices = self.index.search(query_reshaped, k)

            # Extract results
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
                if idx >= 0:  # FAISS returns -1 for invalid indices
                    results.append({
                        "metadata": self.metadata[idx].copy(),
                        "score": float(score),
                        "rank": rank,
                        "index": int(idx),
                    })

            logger.debug(f"Search completed: returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def save_index(self, index_name: str = "default") -> Path:
        """
        Save the FAISS index and metadata to disk.

        Saves both the FAISS index binary file and a JSON metadata file
        to the index folder. Creates a subfolder for the named index.

        Args:
            index_name: Name for the index (default: "default").
                       Determines subfolder in indexes/ directory.

        Returns:
            Path to the saved index directory.

        Raises:
            RuntimeError: If index hasn't been created or is empty.
            Exception: If save operation fails.

        Example:
            >>> vs = VectorStore()
            >>> vs.create_index()
            >>> # ... add embeddings ...
            >>> path = vs.save_index("paper_embeddings")
            >>> print(f"Saved to {path}")
        """
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        if self.index.ntotal == 0:
            logger.warning("Saving empty index (0 vectors)")

        try:
            # Create index-specific folder
            index_path = self.index_folder / index_name
            index_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_file = index_path / INDEX_FILE_NAME
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index: {index_file}")

            # Save metadata as JSON
            metadata_file = index_path / METADATA_FILE_NAME
            with open(metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata: {metadata_file}")

            logger.info(
                f"Index '{index_name}' saved with {self.index.ntotal} vectors"
            )
            return index_path

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, index_name: str = "default") -> bool:
        """
        Load a FAISS index and metadata from disk.

        Loads both the FAISS index binary file and the associated metadata
        JSON file from the specified folder.

        Args:
            index_name: Name of the index to load (default: "default").
                       Determines subfolder in indexes/ directory.

        Returns:
            True if load successful, False if index doesn't exist.

        Raises:
            Exception: If load operation fails or files are corrupted.

        Example:
            >>> vs = VectorStore()
            >>> if vs.load_index("paper_embeddings"):
            ...     print(f"Loaded {vs.index.ntotal} vectors")
            ... else:
            ...     print("Index not found")
        """
        index_path = self.index_folder / index_name

        if not index_path.exists():
            logger.warning(f"Index folder not found: {index_path}")
            return False

        try:
            # Load FAISS index
            index_file = index_path / INDEX_FILE_NAME
            if not index_file.exists():
                logger.warning(f"FAISS index file not found: {index_file}")
                return False

            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index: {index_file}")

            # Load metadata
            metadata_file = index_path / METADATA_FILE_NAME
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}")
                self.metadata = []
            else:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata: {metadata_file}")

            logger.info(
                f"Index '{index_name}' loaded with {self.index.ntotal} vectors"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.

        Returns:
            Dictionary containing index statistics and configuration.

        Example:
            >>> info = vs.get_index_info()
            >>> print(f"Index has {info['total_vectors']} vectors")
        """
        if self.index is None:
            return {
                "status": "uninitialized",
                "dimension": self.dimension,
            }

        return {
            "status": "initialized",
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "metadata_count": len(self.metadata),
            "index_type": type(self.index).__name__,
            "index_folder": str(self.index_folder),
        }

    def reset(self) -> None:
        """
        Clear the index and metadata.

        Removes all vectors and metadata from memory. Use before loading
        a different index or starting fresh.

        Example:
            >>> vs.reset()
            >>> print(f"Index cleared: {vs.index.ntotal} vectors remaining")
        """
        self.index = None
        self.metadata = []
        logger.info("Vector store reset")

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            Number of embedding vectors in the index.

        Example:
            >>> count = vs.get_vector_count()
            >>> print(f"Index contains {count} vectors")
        """
        return self.index.ntotal if self.index is not None else 0

    def list_saved_indexes(self) -> List[str]:
        """
        List all saved index names in the indexes folder.

        Returns:
            List of index names (subdirectory names).

        Example:
            >>> indexes = vs.list_saved_indexes()
            >>> print(f"Available indexes: {indexes}")
        """
        if not self.index_folder.exists():
            return []

        indexes = [
            d.name
            for d in self.index_folder.iterdir()
            if d.is_dir() and (d / INDEX_FILE_NAME).exists()
        ]
        return sorted(indexes)

    def delete_index(self, index_name: str) -> bool:
        """
        Delete a saved index from disk.

        Removes the index folder and all associated files.

        Args:
            index_name: Name of the index to delete.

        Returns:
            True if deletion successful, False if index doesn't exist.

        Example:
            >>> success = vs.delete_index("old_embeddings")
            >>> print(f"Deleted: {success}")
        """
        index_path = self.index_folder / index_name

        if not index_path.exists():
            logger.warning(f"Index not found: {index_path}")
            return False

        try:
            import shutil
            shutil.rmtree(index_path)
            logger.info(f"Deleted index: {index_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            raise


# Convenience functions for module-level usage

def create_vectorstore(
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    index_folder: str = DEFAULT_INDEXES_FOLDER,
) -> VectorStore:
    """
    Create a new vector store instance.

    Args:
        dimension: Embedding dimension (default: 384).
        index_folder: Folder for saving indexes (default: "indexes").

    Returns:
        Initialized VectorStore instance.

    Example:
        >>> vs = create_vectorstore()
        >>> vs.create_index()
    """
    vs = VectorStore(dimension=dimension, index_folder=index_folder)
    return vs


def load_vectorstore(
    index_name: str = "default",
    index_folder: str = DEFAULT_INDEXES_FOLDER,
) -> Optional[VectorStore]:
    """
    Load an existing vector store from disk.

    Args:
        index_name: Name of the index to load (default: "default").
        index_folder: Folder containing indexes (default: "indexes").

    Returns:
        Loaded VectorStore instance, or None if load failed.

    Example:
        >>> vs = load_vectorstore("paper_embeddings")
        >>> if vs:
        ...     results = vs.search(query_embedding)
    """
    vs = VectorStore(index_folder=index_folder)
    if vs.load_index(index_name):
        return vs
    return None
