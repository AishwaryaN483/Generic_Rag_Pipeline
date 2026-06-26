import os
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

parsed_file = os.path.join(
    OUTPUT_DIR,
    "parsed_documents.pkl"
)

chunks_file = os.path.join(
    OUTPUT_DIR,
    "chunks.pkl"
)

# ---------------- LOAD PARSED DOCUMENTS ----------------

with open(parsed_file, "rb") as f:
    documents = pickle.load(f)

print("=" * 70)
print("PARSED DOCUMENTS LOADED")
print("=" * 70)

print(f"Documents Loaded : {len(documents)}")

# ---------------- CHUNKING ----------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_documents(documents)

# ---------------- SAVE CHUNKS ----------------

with open(chunks_file, "wb") as f:
    pickle.dump(chunks, f)

# ---------------- DISPLAY ----------------

print("\n")
print("=" * 70)
print("CHUNKING COMPLETED")
print("=" * 70)

print(f"Total Chunks Created : {len(chunks)}")

for i, chunk in enumerate(chunks[:5], start=1):

    print(f"\nChunk {i}")

    print(f"Characters : {len(chunk.page_content)}")

    print("Metadata :", chunk.metadata)

    print("\nPreview:\n")

    print(chunk.page_content[:250])

    print("\n" + "-" * 70)

print(f"\nChunks saved to:\n{chunks_file}")