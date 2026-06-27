import os
import pickle
import faiss
import numpy as np

# ---------------- PATHS ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

embedding_file = os.path.join(
    OUTPUT_DIR,
    "embedding_data.pkl"
)

index_file = os.path.join(
    OUTPUT_DIR,
    "faiss_index.index"
)

chunk_store_file = os.path.join(
    OUTPUT_DIR,
    "chunk_store.pkl"
)

# ---------------- LOAD EMBEDDINGS ---------------- #

with open(embedding_file, "rb") as f:

    data = pickle.load(f)

chunks = data["chunks"]

embeddings = data["embeddings"]

print("="*70)
print("EMBEDDINGS LOADED")
print("="*70)

print(f"Chunks : {len(chunks)}")
print(f"Embeddings : {len(embeddings)}")

# ---------------- CONVERT TO NUMPY ---------------- #

embedding_matrix = np.array(
    embeddings,
    dtype="float32"
)

print("\nEmbedding Matrix Shape :", embedding_matrix.shape)

# ---------------- CREATE FAISS INDEX ---------------- #

dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_matrix)

# ---------------- SAVE ---------------- #

faiss.write_index(index, index_file)

# ---------------- SAVE CHUNK STORE ---------------- #

with open(chunk_store_file, "wb") as f:

    pickle.dump(chunks, f)

print("\n")
print("="*70)
print("FAISS INDEX CREATED")
print("="*70)

print(f"Vectors Stored : {index.ntotal}")

print(f"Embedding Dimension : {dimension}")

print(f"\nFAISS Index Saved To :")
print(index_file)

print(f"\nChunk Store Saved To :")
print(chunk_store_file)