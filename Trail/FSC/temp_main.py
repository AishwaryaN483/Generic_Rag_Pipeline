# This QA model uses Flan-T5-base model which is the plain flan model that does basic instruction following model
# it doesnt retrive text it just copies most relevant in the input text  ( NO understanding of CONCEPTUAL)
# --------------need a fine-tuning model like "google/flan-t5-large"

import sys
import time
import tracemalloc

# ---------- Import Custom Parsers ----------
sys.path.append("parsers")
from Trail.FSC.src.Parsing.parsepdf import parse_pdf
from Trail.FSC.src.Parsing.parsedocx import parse_docx
from Trail.FSC.src.Parsing.parseppt import parse_pptx
from visualize_embeddings import visualize_embeddings_tsne
from embedding import embed_chunks  # make sure query embedding is computed
from visualize_embeddings import visualize_embeddings_tsne_with_query
from QA_model2 import get_best_answer   # ✅ Added import

from faiss_indexer import build_faiss_index, search_faiss


# ---------- Benchmark Wrapper ----------
def benchmark(func, *args, **kwargs):
    start = time.perf_counter()
    tracemalloc.start()

    result = func(*args, **kwargs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end = time.perf_counter()

    stats = {
        "time": end - start,
        "current_mem": current / 10**6,
        "peak_mem": peak / 10**6,
    }

    return result, stats


# ---------- Chunking ----------
def chunk_text_by_words(text, chunk_size=200, overlap=0):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ---------- Main ----------
def main():
    print("Select file type:")
    print("1. PDF")
    print("2. DOCX")
    print("3. PPTX")
    choice = input("Enter 1, 2, or 3: ").strip()

    file_path = input("Paste the full path of the file: ").strip()

    # ---- Parse ----
    if choice == "1":
        text, parse_stats = benchmark(parse_pdf, file_path)
    elif choice == "2":
        text, parse_stats = benchmark(parse_docx, file_path)
    elif choice == "3":
        text, parse_stats = benchmark(parse_pptx, file_path)
    else:
        print("❌ Invalid choice")
        return

    print(f"\n Parsing -> Time: {parse_stats['time']:.4f}s | "
          f"Memory: {parse_stats['current_mem']:.2f}MB (Peak {parse_stats['peak_mem']:.2f}MB)")

    # ---- Chunk ----
    chunks, chunk_stats = benchmark(chunk_text_by_words, text, 150, 20)
    print(f" Chunking -> Time: {chunk_stats['time']:.4f}s | "
          f"Memory: {chunk_stats['current_mem']:.2f}MB (Peak {chunk_stats['peak_mem']:.2f}MB)")
    print(f" Total chunks created: {len(chunks)}")

    # ---- Build FAISS Index ----
    index, embeddings = build_faiss_index(chunks)
    print(f"✅ FAISS index created with {index.ntotal} embeddings")

    # ---- Visualize All Chunks ----
    visualize_embeddings_tsne(embeddings, chunks)

    # ---- Query Loop ----
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        results = search_faiss(index, query, chunks, top_k=3)

        print("\n🔎 Retrieved Chunks:")
        for r in results:
            print(f"- {r['chunk'][:80]}... (distance={r['distance']:.4f})")

        # ---- Get Answer ----
        qa_output = get_best_answer(query, results)

        print("\n🧠 FINAL ANSWER:")
        print(f"👉 {qa_output['answer']}")
        print(f"(confidence={qa_output['score']:.4f})")


if __name__ == "__main__":
    main()
