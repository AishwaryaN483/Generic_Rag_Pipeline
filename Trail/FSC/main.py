from operator import index
import sys
import time
import tracemalloc

# ---------- Import Custom Parsers ----------
sys.path.append("parsers")
from Trail.FSC.src.Parsing.parsepdf import parse_pdf
from Trail.FSC.src.Parsing.parsedocx import parse_docx
from Trail.FSC.src.Parsing.parseppt import parse_pptx

# ---------- Import Chunking ----------
from benchmark_chunking import run_all_chunkers

# ---------- Import Other Modules ----------
from visualize_embeddings import visualize_embeddings_tsne
from embedding import embed_chunks_with_model
from visualize_embeddings import visualize_embeddings_tsne_with_query
from qa_model import get_best_answer_extractive, get_best_answer_generative
from faiss_indexer import build_faiss_index, search_faiss
from reranker import rerank_results


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
        parsed_output, parse_stats = benchmark(parse_pdf, file_path)
    elif choice == "2":
        parsed_output, parse_stats = benchmark(parse_docx, file_path)
    elif choice == "3":
        parsed_output, parse_stats = benchmark(parse_pptx, file_path)
    else:
        print("❌ Invalid choice")
        return

    text = parsed_output["text"]
    tables = parsed_output["tables"]
    print(f"\n Parsing -> Time: {parse_stats['time']:.4f}s | "
          f"Memory: {parse_stats['current_mem']:.2f}MB (Peak {parse_stats['peak_mem']:.2f}MB)")
    print(f"🧾 Extracted {len(tables)} tables separately.")

    # ---- Chunking ----
    print("\n🚀 Running All Chunking Strategies...")
    chunk_results = run_all_chunkers(text, tables)
    for name, data in chunk_results.items():
        s = data["stats"]
        print(f"\n🔹 {name} Chunking:")
        print(f"   Time: {s['time']:.4f}s | Memory: {s['current_mem']:.2f}MB (Peak {s['peak_mem']:.2f}MB)")
        print(f"   Total chunks: {data['count']}")

    selected_strategy = input("\nSelect strategy (Words / Sentences / Paragraphs / Semantic / Topic): ").strip()
    chunks = chunk_results[selected_strategy]["chunks"]

    # ---- Embedding Model Selection ----
    print("\n🔹 Select Embedding Model:")
    print("1. intfloat/e5-base-v2 (~100M)")
    print("2. sentence-transformers/all-mpnet-base-v2 (~100M)")
    print("3. ibm-granite/granite-embedding-small-english-r2 (~47M)")
    print("4. ibm-granite/granite-embedding-english-r2 (~149M)")
    emb_choice = input("Enter 1, 2, 3, or 4: ").strip()

    if emb_choice == "1":
        emb_model = "intfloat/e5-base-v2"
    elif emb_choice == "2":
        emb_model = "sentence-transformers/all-mpnet-base-v2"
    elif emb_choice == "3":
        emb_model = "ibm-granite/granite-embedding-small-english-r2"
    elif emb_choice == "4":
        emb_model = "ibm-granite/granite-embedding-english-r2"
    else:
        print("❌ Invalid embedding model.")
        return

    # ---- Embed and Build Index ----
    embeddings, embed_stats = benchmark(embed_chunks_with_model, chunks, emb_model)

    index = build_faiss_index(embeddings)

    print(f"✅ FAISS index created with {index.ntotal} embeddings")
    print(f"Embedding Time: {embed_stats['time']:.2f}s | Memory: {embed_stats['peak_mem']:.2f}MB")


    visualize_embeddings_tsne(embeddings, chunks)

    # ---- QA Model Selection ----
    print("\n🧠 Select QA Model:")
    print("1. Extractive (roberta-base-squad2)")
    print("2. Generative (distilbert/distilgpt2)")
    print("3. Generative (HuggingFaceTB/SmolLM2-135M-Instruct)")
    model_choice = input("Enter 1, 2, or 3: ").strip()
    if model_choice == "1":
        model_type = "extractive"
    elif model_choice == "2":
        model_type = "gpt2"
    elif model_choice == "3":
        model_type = "smollm"
    else:
        print("❌ Invalid QA model choice")
        return

    # ---- Query Loop ----
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        # Embed query
        query_embedding = embed_chunks_with_model([query], emb_model)

        # FAISS search
        indices, distances = search_faiss(index, query_embedding, top_k=5)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "chunk": chunks[idx],
                "score": float(dist)
            })

        print(f"\n🔎 Retrieved top {len(results)} chunks. Re-ranking...")

        reranked = rerank_results(query, results, emb_model)
        top_results = reranked[:3]

        for r in top_results:
            print(f"- {r['chunk'][:80]}... (rerank_score={r['score']:.4f})")

        if model_type == "extractive":
            qa_output = get_best_answer_extractive(query, top_results, model_type)
        else:
            qa_output = get_best_answer_generative(query, top_results, model_type)

        print("\n🧠 FINAL ANSWER:")
        print(f"👉 {qa_output['answer']}")
        score = qa_output.get("score")
        print(f"(confidence={score:.4f})" if isinstance(score, float) else "(confidence=not available)")


if __name__ == "__main__":
    main()
