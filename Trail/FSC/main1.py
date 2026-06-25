import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from Trail.FSC.src.Chunking.chunk_paragraph import content_aware_chunk
from Trail.FSC.src.Chunking.chunk_sentence import chunk_text_by_sentences
from chunk_words import chunk_text_by_words
from chunk_proposition import extract_propositions
from chunk_semantic import chunk_text_semantically
from Trail.FSC.src.Parsing.parsepdf import parse_pdf


# ============================================================
# CLEAN TEXT (PDF SAFE)
# ============================================================
def clean_text(text: str) -> str:
    text = text.replace("", "-").replace("•", "-")
    text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
    text = re.sub(r'\b(Abstract|Introduction|Background|Conclusion)\s+\1\b',
                  r'\1', text, flags=re.I)
    text = re.sub(r'(?<!\n)\n(?=[a-z])', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(Input-Input Layer\d+.*?Figure\s*\d+:)', '', text, flags=re.S)
    text = re.sub(r'>SOE<|>dap<', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
# PRINT CHUNKS
# ============================================================
def print_chunks(title, chunks):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk #{i} ---")
        print(chunk.strip())


# ============================================================
# COVERAGE (RAG-SAFE, WORD-SET BASED)
# ============================================================
def coverage_report(original_text, chunks):
    original_words = set(original_text.lower().split())
    chunk_words = set(" ".join(chunks).lower().split())

    coverage = round(
        (len(original_words & chunk_words) / max(len(original_words), 1)) * 100,
        2
    )

    sizes = [len(c.split()) for c in chunks if c.strip()]

    return {
        "coverage": coverage,
        "chunks": len(chunks),
        "avg": sum(sizes) // max(len(sizes), 1),
        "max": max(sizes, default=0),
        "min": min(sizes, default=0)
    }


def print_coverage(label, r):
    print(f"\n📊 {label} COVERAGE")
    print("-" * 30)
    print(f"Coverage : {r['coverage']}%")
    print(f"Chunks   : {r['chunks']}")
    print(f"Avg size : {r['avg']} words")
    print(f"Max size : {r['max']} words")
    print(f"Min size : {r['min']} words")


# ============================================================
# EMBEDDING
# ============================================================
def load_embedding_model(choice):
    models = {
        "1": "intfloat/e5-base-v2",
        "2": "sentence-transformers/all-mpnet-base-v2",
        "3": "ibm-granite/granite-embedding-small-english-r2",
        "4": "ibm-granite/granite-embedding-english-r2"
    }
    if choice not in models:
        raise ValueError("Invalid embedding choice")
    return SentenceTransformer(models[choice])


def build_faiss(chunks, model):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve(query, chunks, model, index, top_k=5):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, top_k)
    return [chunks[i] for i in ids[0]]


# ============================================================
# QA MODELS
# ============================================================
def load_qa_model(choice):
    if choice == "1":
        return pipeline("question-answering",
                        model="deepset/roberta-base-squad2"), "extractive"
    if choice == "2":
        return pipeline("text-generation",
                        model="distilgpt2",
                        max_new_tokens=150), "generative"
    if choice == "3":
        return pipeline("text-generation",
                        model="HuggingFaceTB/SmolLM2-135M-Instruct",
                        max_new_tokens=150), "generative"
    raise ValueError("Invalid QA choice")


def answer_question(query, contexts, qa_model, mode):
    context = "\n\n".join(contexts[:3])
    if mode == "extractive":
        return qa_model(question=query, context=context)["answer"]
    else:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return qa_model(prompt)[0]["generated_text"]


# ============================================================
# MAIN
# ============================================================
def main():
    path = input("Paste PDF path: ").strip()
    if not os.path.exists(path):
        print("❌ File not found")
        return

    raw = parse_pdf(path)
    text = clean_text(raw)

    print("\n🔹 Select Chunking Method:")
    print("1. Paragraph")
    print("2. Sentence")
    print("3. Word")
    print("4. Proposition")
    print("5. Semantic")

    c = input("Choice: ").strip()

    if c == "1":
        chunks = content_aware_chunk(text)
        label = "Paragraph"
    elif c == "2":
        chunks = chunk_text_by_sentences(text, 2)
        label = "Sentence"
    elif c == "3":
        chunks = chunk_text_by_words(text, 200, 50)
        label = "Word"
    elif c == "4":
        chunks = extract_propositions(text)
        label = "Proposition"
    elif c == "5":
        s = chunk_text_by_sentences(text, 2)
        chunks = chunk_text_semantically(" ".join(s))
        label = "Semantic"
    else:
        print("❌ Invalid choice")
        return

    print_chunks(f"{label.upper()} CHUNKS", chunks)
    print_coverage(label, coverage_report(text, chunks))

    # ------------------ EMBEDDING ------------------
    print("\n🔹 Select Embedding Model:")
    print("1. e5-base-v2")
    print("2. all-mpnet-base-v2")
    print("3. granite-small")
    print("4. granite-large")

    emb_choice = input("Choice: ").strip()
    embedder = load_embedding_model(emb_choice)
    index = build_faiss(chunks, embedder)

    print(f"✅ FAISS index built with {index.ntotal} chunks")

    # ------------------ QA ------------------
    print("\n🧠 Select QA Model:")
    print("1. Extractive (RoBERTa)")
    print("2. Generative (GPT-2)")
    print("3. Generative (SmolLM)")

    qa_choice = input("Choice: ").strip()
    qa_model, mode = load_qa_model(qa_choice)

    # ------------------ QUERY LOOP ------------------
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        ctx = retrieve(q, chunks, embedder, index)
        ans = answer_question(q, ctx, qa_model, mode)

        print("\n🧠 ANSWER:")
        print(ans)


# ============================================================
if __name__ == "__main__":
    main()
