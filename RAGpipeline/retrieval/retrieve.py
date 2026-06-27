import os
import pickle
import faiss
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- PATHS ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index.index")
CHUNK_FILE = os.path.join(OUTPUT_DIR, "chunk_store.pkl")

# ---------------- LOAD FAISS ---------------- #

index = faiss.read_index(INDEX_FILE)

with open(CHUNK_FILE, "rb") as f:
    chunks = pickle.load(f)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# Retrieval Function
# ==========================================================

def retrieve_documents(query, top_k=3):

    query_embedding = embedding_model.embed_query(query)

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ==========================================================
# Test Retrieval
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FAISS INDEX LOADED")
    print("=" * 70)

    print(f"Total Stored Vectors : {index.ntotal}")
    print(f"Total Chunks : {len(chunks)}")

    question = input("\nEnter Question : ")

    results = retrieve_documents(question)

    print("\n")
    print("=" * 70)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 70)

    for i, chunk in enumerate(results, start=1):

        print(f"\nRank {i}")

        print(chunk.metadata)

        print()

        print(chunk.page_content)

        print("\n" + "-" * 70)