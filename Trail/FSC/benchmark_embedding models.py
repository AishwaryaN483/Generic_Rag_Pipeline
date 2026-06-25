import time
import tracemalloc
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------------------------------------
# Models to benchmark
# -----------------------------------------------------------
EMBEDDING_MODELS = {
    "intfloat/e5-base-v2": "E5 Base",
    "sentence-transformers/all-mpnet-base-v2": "MPNet Base",
    "ibm-granite/granite-embedding-small-english-r2": "Granite Small",
    "ibm-granite/granite-embedding-english-r2": "Granite Base",
}

# -----------------------------------------------------------
# Test Data (Replace with your actual chunks)
# -----------------------------------------------------------
TEST_TEXT = [
    "TopicTrace is an AI-powered RAG-based system for document understanding.",
    "The embedding models are used to convert text chunks into dense vector representations.",
    "FAISS is employed for similarity search and fast retrieval operations.",
    "Chunking strategies include words, sentences, paragraphs, semantic clusters, and topic modeling.",
] * 50   # → Makes ~200 chunks for realistic testing


# -----------------------------------------------------------
# Benchmark Helper
# -----------------------------------------------------------
def benchmark_embedding(model_name, chunks):
    print(f"\n🔍 Benchmarking Model: {model_name}")

    # ------------------------------
    # Load model timing
    # ------------------------------
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - t0

    # ------------------------------
    # Embedding time + memory
    # ------------------------------
    tracemalloc.start()
    t1 = time.perf_counter()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    t2 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert memory to MB
    current_mb = current / (1024 * 1024)
    peak_mb = peak / (1024 * 1024)

    # ------------------------------
    # Stats
    # ------------------------------
    total_time = t2 - t1
    dim = embeddings.shape[1]
    throughput = len(chunks) / total_time  # chunks per second

    result = {
        "model": model_name,
        "load_time": load_time,
        "embedding_time": total_time,
        "current_memory_mb": round(current_mb, 2),
        "peak_memory_mb": round(peak_mb, 2),
        "embedding_dim": dim,
        "throughput_chunks_per_sec": round(throughput, 2)
    }

    return result


# -----------------------------------------------------------
# Main Runner
# -----------------------------------------------------------
def main():
    results = []

    print("🚀 Starting Embedding Benchmark...\n")
    print(f"Total test chunks: {len(TEST_TEXT)}")

    for model_path, display_name in EMBEDDING_MODELS.items():
        res = benchmark_embedding(model_path, TEST_TEXT)
        results.append(res)

    # Print table
    print("\n\n==================== Benchmark Results ====================")
    print(f"{'Model':30} | {'Embed Time (s)':>15} | {'Peak Mem (MB)':>12} | {'Load Time (s)':>12} | {'Dim':>6} | {'Chunks/Sec':>12}")
    print("-" * 100)

    for r in results:
        print(f"{r['model'][:30]:30} | "
              f"{r['embedding_time']:15.2f} | "
              f"{r['peak_memory_mb']:12.2f} | "
              f"{r['load_time']:12.2f} | "
              f"{r['embedding_dim']:6} | "
              f"{r['throughput_chunks_per_sec']:12.2f}")

    print("\n⚡ Benchmark complete!\n")

if __name__ == "__main__":
    main()
