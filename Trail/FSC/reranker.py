import numpy as np
from sentence_transformers import SentenceTransformer, util

def rerank_results(query, results, model_name):
    """Re-rank retrieved chunks based on query similarity."""
    model = SentenceTransformer(model_name)
    query_emb = model.encode(query, convert_to_tensor=True)
    chunks = [r["chunk"] for r in results]
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, chunk_embs)[0]
    for i, s in enumerate(scores):
        results[i]["score"] = float(s)
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results
