from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text_semantically(text, threshold=0.55):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return []

    if len(sentences) == 1:
        return [sentences[0] + "."]

    embeddings = MODEL.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()

        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks
