# ---------- faiss_indexer.py ----------
import faiss
import numpy as np


# ======================================================
# Build FAISS index (FAST, BULK MODE)
# ======================================================
def build_faiss_index(embeddings):
    """
    Build a FAISS index for given embeddings.
    embeddings: list or numpy array of float32 vectors
    Returns:
        index: built FAISS index
        embeddings: float32 ndarray
    """
    print("\n⚡ Building FAISS index...")

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    # Create FAISS L2 index
    index = faiss.IndexFlatL2(dim)

    # Add all vectors at once (fast!)
    index.add(embeddings)

    print(f"✅ Index built successfully!")
    print(f"   → Total vectors: {index.ntotal}")
    print(f"   → Embedding dimension: {dim}")

    return index


# ======================================================
# Search FAISS index
# ======================================================
def search_faiss(index, query_embedding, top_k=3):
    """
    Search FAISS index for nearest neighbors.

    query_embedding: 2D np.array of shape (1, dim)
    returns:
        indices, distances
    """
    query_embedding = np.array(query_embedding).astype("float32")

    # Safety check
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"❌ Dimension mismatch: query dim={query_embedding.shape[1]}, index dim={index.d}. "
            "Use the same embedding model."
        )

    distances, indices = index.search(query_embedding, top_k)
    return indices, distances
