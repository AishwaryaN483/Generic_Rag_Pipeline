import time
import tracemalloc
import torch
from openpyxl import Workbook

# ---------- YOUR MODULES ----------
from benchmark_chunking import run_all_chunkers
from embedding import embed_chunks_with_model
from faiss_indexer import build_faiss_index, search_faiss
from reranker import rerank_results
from qa_model import get_best_answer_extractive, get_best_answer_generative
from Trail.FSC.src.Parsing.parsepdf import parse_pdf

# ---------- CONFIG ----------
CHUNKING_STRATEGIES = [ "Topic"]

EMBEDDING_MODELS = {
    "E5": "intfloat/e5-base-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "Granite-Small": "ibm-granite/granite-embedding-small-english-r2",
    "Granite-Large": "ibm-granite/granite-embedding-english-r2",
}

QA_MODELS = {
    "RoBERTa": "extractive",
    "DistilGPT2": "gpt2",
    "SmolLM": "smollm",
}

QUESTIONS = {
    "Q1": "What is Artificial Intelligence?",
    "Q2": "What are the goals of Artificial Intelligence?",
    "Q3": "What is an intelligent agent?",
    "Q4": "What is fuzzy logic?",
    "Q5": "What are the components of NLP?"
}

TOTAL_RUNS = len(CHUNKING_STRATEGIES) * len(EMBEDDING_MODELS) * len(QA_MODELS)

# ---------- HELPERS ----------
def is_correct(answer: str) -> bool:
    return len(answer.strip()) > 20


def print_header(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70, flush=True)


# ---------- MAIN BENCHMARK ----------
def run_benchmark(pdf_path):

    print_header("📄 Parsing Document")
    parsed = parse_pdf(pdf_path)
    text, tables = parsed["text"], parsed["tables"]

    print("✔ Document parsed successfully\n", flush=True)

    print_header("🧩 Running Chunking Strategies")
    chunked = run_all_chunkers(text, tables)

    # ---------- Excel ----------
    wb = Workbook()
    ws = wb.active
    ws.title = "RAG Benchmark"

    ws.append([
        "Chunking", "Embedding", "QA Model",
        "Avg Time (s)", "Peak Memory (MB)",
        "Q1", "Q2", "Q3", "Q4", "Q5",
        "Accuracy (%)"
    ])

    run_counter = 0
    global_start = time.time()

    # ---------- BENCH LOOP ----------
    for chunking in CHUNKING_STRATEGIES:
        chunks = chunked[chunking]["chunks"]

        print_header(f"🔹 Chunking: {chunking} ({len(chunks)} chunks)")

        for emb_name, emb_model in EMBEDDING_MODELS.items():

            print(f"⚡ Embedding with {emb_name} ...", flush=True)
            tracemalloc.start()
            t0 = time.perf_counter()
            embeddings = embed_chunks_with_model(chunks, emb_model)
            embed_time = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mem = peak_mem / 1e6

            print(f"   ✔ Embeddings done in {embed_time:.2f}s | "
                  f"Peak Mem {peak_mem:.0f} MB", flush=True)

            print("⚡ Building FAISS index...", flush=True)
            index = build_faiss_index(embeddings)
            print(f"   ✔ Index built (vectors={index.ntotal})\n", flush=True)

            for qa_name, qa_type in QA_MODELS.items():

                run_counter += 1
                print(
                    f"[{run_counter}/{TOTAL_RUNS}] ▶ Running: "
                    f"{chunking} | {emb_name} | {qa_name}",
                    flush=True
                )

                results = {}
                total_time = 0.0

                for qid, question in QUESTIONS.items():
                    q_start = time.perf_counter()

                    query_emb = embed_chunks_with_model([question], emb_model)
                    I, _ = search_faiss(index, query_emb, top_k=5)

                    retrieved = [{"chunk": chunks[i]} for i in I[0]]
                    reranked = rerank_results(question, retrieved, emb_model)[:3]

                    if qa_type == "extractive":
                        out = get_best_answer_extractive(question, reranked, qa_type)
                    else:
                        out = get_best_answer_generative(question, reranked, qa_type)

                    q_time = time.perf_counter() - q_start
                    total_time += q_time
                    correct = is_correct(out["answer"])
                    results[qid] = correct

                    status = "✔" if correct else "✖"

                    print(f"    {qid} {status} ({q_time:.2f}s)", flush=True)
                    print(f"        Answer: {out['answer']}\n", flush=True)


                avg_time = total_time / len(QUESTIONS)
                accuracy = (sum(results.values()) / len(results)) * 100

                ws.append([
                    chunking,
                    emb_name,
                    qa_name,
                    round(avg_time, 3),
                    round(peak_mem, 2),
                    *["✔" if results[q] else "✖" for q in QUESTIONS],
                    round(accuracy, 1)
                ])

                elapsed = time.time() - global_start
                avg_run = elapsed / run_counter
                remaining = TOTAL_RUNS - run_counter
                eta = remaining * avg_run / 60

                print(
                    f"✔ Finished: {chunking} | {emb_name} | {qa_name} | "
                    f"avg={avg_time:.2f}s | acc={accuracy:.1f}%",
                    flush=True
                )
                print(f"⏳ ETA: ~{eta:.1f} minutes remaining", flush=True)
                print("-" * 70, flush=True)

    wb.save("rag_full_60_combo_benchmark.xlsx")
    print_header("✅ BENCHMARK COMPLETE")
    print("📊 Results saved to rag_full_60_combo_benchmark.xlsx", flush=True)


# ---------- RUN ----------
if __name__ == "__main__":

    PDF_PATH = r"C:\Users\bhuwa\topicTrace\Trail\FSC\artificial_intelligence_tutorial.pdf"
    run_benchmark(PDF_PATH)
