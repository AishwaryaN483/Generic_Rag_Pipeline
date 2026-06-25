import time
import tracemalloc
from chunk_words import chunk_text_by_words
from Trail.FSC.src.Chunking.chunk_sentence import chunk_text_by_sentences
from Trail.FSC.src.Chunking.chunk_paragraph import chunk_text_by_paragraphs
from chunk_semantic import chunk_text_semantically
from chunk_topic import chunk_topic

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


def run_all_chunkers(text, tables):
    strategies = {
        "Words": (chunk_text_by_words, {"chunk_size": 150, "overlap": 20}),
        "Sentences": (chunk_text_by_sentences, {"max_sentences": 5}),
        "Paragraphs": (chunk_text_by_paragraphs, {}),
        "Semantic": (chunk_text_semantically, {"threshold": 0.5}),
        "Topic": (chunk_topic, {"num_topics": 3}),
    }

    results = {}
    for name, (func, params) in strategies.items():
        chunks, stats = benchmark(func, text, **params)

        # Add tables as separate chunks
        all_chunks = chunks + tables

        results[name] = {
            "chunks": all_chunks,
            "stats": stats,
            "count": len(all_chunks),
        }

    return results
