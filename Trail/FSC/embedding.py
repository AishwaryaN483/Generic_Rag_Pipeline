from sentence_transformers import SentenceTransformer
import torch

_MODEL_CACHE = {}

def embed_chunks_with_model(chunks, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(
            model_name,
            device=device
        )

    model = _MODEL_CACHE[model_name]

    embeddings = model.encode(
        chunks,
        batch_size=32,              # IMPORTANT for GPU
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings
