from Trail.FSC.Chunking.chunk_paragraph import content_aware_chunk

def chunk_text_by_sentences(text, max_sentences=5):
    paragraphs = content_aware_chunk(text)
    chunks = []

    for para in paragraphs:
        # Keep lists atomic
        if para.strip().startswith(("-", "1.")):
            chunks.append(para)
            continue

        sentences = [s.strip() for s in para.split(".") if s.strip()]
        for i in range(0, len(sentences), max_sentences):
            chunks.append(". ".join(sentences[i:i + max_sentences]) + ".")

    return chunks
