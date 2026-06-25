from Trail.FSC.src.Chunking.chunk_paragraph import content_aware_chunk
from Trail.FSC.src.Chunking.chunk_sentence import chunk_text_by_sentences


def final_chunk(text):
    # Step 1: Paragraph-aware chunking
    base_chunks = content_aware_chunk(text)

    final_chunks = []

    for chunk in base_chunks:
        # Step 2: refine large chunks
        if len(chunk.split()) > 300:
            refined = chunk_text_by_sentences(chunk)
            final_chunks.extend(refined)
        else:
            final_chunks.append(chunk)

    return final_chunks

chunks = final_chunk(text)

print("Total chunks:", len(chunks))

for i, c in enumerate(chunks[:5]):
    print(f"\nChunk {i}:")
    print(c)
    print("Length:", len(c.split()))