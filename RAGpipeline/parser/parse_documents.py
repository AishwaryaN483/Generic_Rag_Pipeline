import os
import pickle

from pdf_parser import parse_pdf
from docx_parser import parse_docx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

documents = []

# ---------------- PDF ----------------

pdf_folder = os.path.join(DATA_DIR, "pdf")

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        file_path = os.path.join(pdf_folder, file)

        documents.extend(
            parse_pdf(file_path)
        )

# ---------------- DOCX ----------------

docx_folder = os.path.join(DATA_DIR, "docx")

for file in os.listdir(docx_folder):

    if file.endswith(".docx"):

        file_path = os.path.join(docx_folder, file)

        documents.extend(
            parse_docx(file_path)
        )

# ---------------- SAVE ----------------

output_file = os.path.join(
    OUTPUT_DIR,
    "parsed_documents.pkl"
)

with open(output_file, "wb") as f:
    pickle.dump(documents, f)

# ---------------- DISPLAY ----------------

print("=" * 70)
print("DOCUMENT PARSING COMPLETED")
print("=" * 70)

print(f"\nTotal Documents : {len(documents)}")

for i, doc in enumerate(documents, start=1):

    print(f"\nDocument {i}")

    print("Metadata :", doc.metadata)

    print(f"Characters : {len(doc.page_content)}")

    print("\nPreview:\n")

    print(doc.page_content[:300])

    print("\n" + "-" * 70)

print(f"\nParsed documents saved to:\n{output_file}")