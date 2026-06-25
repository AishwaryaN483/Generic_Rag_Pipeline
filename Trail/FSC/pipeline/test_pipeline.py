import os

# Import parsers
from ingestion.parse_pdf import parse_pdf
from ingestion.parse_pptx import parse_pptx
from ingestion.parse_docx import parse_docx

# Import chunker
from chunking.main_chunker import final_chunk


def get_parser(file_path):
    if file_path.endswith(".pdf"):
        return parse_pdf
    elif file_path.endswith(".pptx"):
        return parse_pptx
    elif file_path.endswith(".docx"):
        return parse_docx
    else:
        raise ValueError("Unsupported file format")


def run_pipeline(file_path):
    print(f"\n📄 Processing: {file_path}")

    # Step 1: Parse
    parser = get_parser(file_path)
    parsed_output = parser(file_path)

    text = parsed_output["text"]

    print("\n✅ Parsing Done")
    print("Text length:", len(text))

    # Step 2: Chunking
    chunks = final_chunk(text)

    print("\n✅ Chunking Done")
    print("Total chunks:", len(chunks))

    # Show sample chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
        print("Words:", len(chunk.split()))

    return chunks


if __name__ == "__main__":
    file_path = "../Samples/ppts/Sample1.pptx"  # change this

    run_pipeline(file_path)