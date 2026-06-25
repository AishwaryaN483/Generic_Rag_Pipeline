import time
import csv
import tracemalloc
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# IMPORT YOUR MODULES
# -----------------------------
from chunk_words import chunk_text_by_words
from Trail.FSC.src.Chunking.chunk_sentence import chunk_text_by_sentences
from Trail.FSC.src.Chunking.chunk_paragraph import chunk_text_by_paragraphs
from chunk_semantic import chunk_text_semantically
from chunk_topic import chunk_topic

from embedding import embed_chunks_with_model
from faiss_indexer import build_faiss_index, search_faiss

# -----------------------------
# QUESTIONS
# -----------------------------
QUESTIONS = [
    "What is Artificial Intelligence?",
    "What are the goals of AI?",
    "What is the difference between human and machine intelligence?",
    "What are the components of an intelligent agent?",
    "What is the Turing Test?"
]

# -----------------------------
# CHUNKERS
# -----------------------------
CHUNKERS = {
    "words": lambda text: chunk_text_by_words(text, chunk_size=150, overlap=20),
    "sentences": lambda text: chunk_text_by_sentences(text, max_sentences=5),
    "paragraphs": lambda text: chunk_text_by_paragraphs(text),
    "semantic": lambda text: chunk_text_semantically(text, threshold=0.5),
    "topic": lambda text: chunk_topic(text, num_topics=3),
}

# -----------------------------
# EMBEDDING MODELS
# -----------------------------
EMBED_MODELS = {
    "e5-base-v2": "intfloat/e5-base-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "granite-small": "ibm-granite/granite-embedding-small-english-r2",
    "granite-base": "ibm-granite/granite-embedding-english-r2"
}

# -----------------------------
# QA MODELS
# -----------------------------
extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
generative_qa = pipeline("text-generation", model="distilgpt2")

QA_MODELS = {
    "extractive": extractive_qa,
    "generative": generative_qa
}

# -----------------------------
# TIME + MEMORY WRAPPER
# -----------------------------
def measure(fn, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        "time": round(end - start, 4),
        "mem": round(peak / (1024 * 1024), 4)
    }

# -----------------------------
# PDF TEXT LOADER
# -----------------------------
def load_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# ==============================================================
# MAIN BENCHMARK
# ==============================================================
def benchmark(text):
    results = []

    print("\n==================== STARTING BENCHMARK ====================\n")

    for chunk_name, chunk_fn in CHUNKERS.items():
        print(f"\n🟪 Chunking Strategy: {chunk_name}")
        chunks, c_stats = measure(chunk_fn, text)

        for embed_key, embed_model_path in EMBED_MODELS.items():

            print(f"   🔵 Embedding Model: {embed_key}")

            # 1) Load embedding model for queries
            model = SentenceTransformer(embed_model_path)

            # 2) Generate embeddings (PASS MODEL NAME STRING!)
            embeddings, e_stats = measure(
                embed_chunks_with_model,
                chunks,
                embed_model_path
            )

            # 3) Build FAISS index
            index = build_faiss_index(embeddings)

            for qa_name, qa_model in QA_MODELS.items():
                print(f"      🟢 QA Model: {qa_name}")

                for q in QUESTIONS:

                    # ==============================
                    # RETRIEVAL
                    # ==============================
                    query_vec = model.encode([q]).astype("float32")

                    # Correct FAISS call
                    (distances, indices), r_stats = measure(
                        lambda i, v: i.search(v, 3),
                        index,
                        query_vec
                    )

                    retrieved = [chunks[int(i)] for i in indices[0]]
                    context = " ".join(retrieved)

                    # ==============================
                    # QA
                    # ==============================
                    if qa_name == "extractive":
                        answer, q_stats = measure(
                            qa_model,
                            {"question": q, "context": context}
                        )
                        ans_text = answer["answer"]

                    else:
                        prompt = f"""
You are an AI assistant answering questions using ONLY the retrieved context.
RULES:
- Do NOT use external knowledge.
- If answer is missing, say: "The document does not provide this information."

Context:
{context}

Question: {q}

Answer:
"""
                        output, q_stats = measure(
                            qa_model,
                            prompt,
                            max_length=150,
                            num_return_sequences=1
                        )
                        ans_text = output[0]["generated_text"]

                    # ==============================
                    # TOTAL TIME
                    # ==============================
                    total = round(
                        c_stats["time"]
                        + e_stats["time"]
                        + r_stats["time"]
                        + q_stats["time"],
                        4
                    )

                    # ==============================
                    # CONSOLE OUTPUT BLOCK
                    # ==============================
                    print("\n--------------------------------------------------")
                    print(f"QUESTION: {q}")
                    print(f"Chunking: {chunk_name} | Embedding: {embed_key} | QA: {qa_name}")
                    print(f"Chunk Time: {c_stats['time']}s | Mem: {c_stats['mem']}MB")
                    print(f"Embedding Time: {e_stats['time']}s | Mem: {e_stats['mem']}MB")
                    print(f"Retrieve Time: {r_stats['time']}s")
                    print(f"QA Time: {q_stats['time']}s")
                    print(f"TOTAL Time: {total}s")
                    print(f"ANSWER: {ans_text[:150]}")
                    print("--------------------------------------------------\n")

                    # ==============================
                    # STORE RESULTS FOR CSV
                    # ==============================
                    results.append({
                        "chunking": chunk_name,
                        "embedding": embed_key,
                        "qa": qa_name,
                        "question": q,

                        "chunk_time": c_stats["time"],
                        "chunk_mem": c_stats["mem"],

                        "embed_time": e_stats["time"],
                        "embed_mem": e_stats["mem"],

                        "retrieve_time": r_stats["time"],
                        "qa_time": q_stats["time"],
                        "total_time": total,

                        "answer": ans_text[:200]
                    })

    return results

# ==============================================================
# SAVE TO CSV
# ==============================================================
def save_csv(results, filename="rag_pdf_results.csv"):
    with open(filename, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📁 Saved results to {filename}")

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("🚀 Loading PDF...")
    pdf_path = r"C:\Users\bhuwa\topicTrace\Trail\FSC\Samples\pdfs\artificial_intelligence_tutorial.pdf"
    text = load_pdf_text(pdf_path)

    print("⚙️ Running benchmark...\n")
    results = benchmark(text)

    save_csv(results)

    print("\n==================== BENCHMARK COMPLETE ====================\n")
