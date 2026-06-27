import os
import pickle

from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- PATHS ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

chunks_file = os.path.join(
    OUTPUT_DIR,
    "chunks.pkl"
)

embedding_data_file = os.path.join(
    OUTPUT_DIR,
    "embedding_data.pkl"
)

# ---------------- LOAD CHUNKS ---------------- #

with open(chunks_file, "rb") as f:
    chunks = pickle.load(f)

print("=" * 70)
print("CHUNKS LOADED")
print("=" * 70)

print(f"Total Chunks : {len(chunks)}")

# ---------------- LOAD EMBEDDING MODEL ---------------- #

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding Model Loaded.")

# ---------------- CREATE EMBEDDINGS ---------------- #

texts = [chunk.page_content for chunk in chunks]

embeddings = embedding_model.embed_documents(texts)

# ---------------- SAVE ---------------- #

embedding_data = {

    "chunks": chunks,

    "embeddings": embeddings

}

with open(embedding_data_file, "wb") as f:

    pickle.dump(embedding_data, f)

# ---------------- DISPLAY ---------------- #

print("\n")
print("=" * 70)
print("EMBEDDINGS CREATED")
print("=" * 70)

print(f"Total Embeddings : {len(embeddings)}")

print(f"Embedding Dimension : {len(embeddings[0])}")

print("\nFirst 10 values of First Embedding:\n")

print(embeddings[0][:10])

print(f"\nEmbedding data saved to:\n{embedding_data_file}")