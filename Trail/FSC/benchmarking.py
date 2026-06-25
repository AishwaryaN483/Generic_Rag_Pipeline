import time
import sys
import os

import pdfplumber        # for PDF parsing
import docx              # for DOCX parsing
from pptx import Presentation  # for PPTX parsing

# Import your own functions
sys.path.append("parsers")
from Trail.FSC.src.Parsing.parsepdf import parse_pdf
from Trail.FSC.src.Parsing.parsedocx import parse_docx
from Trail.FSC.src.Parsing.parseppt import parse_pptx
from main import chunk_text_by_words


# ---------- Benchmarking Function ----------
def benchmark_file(file_path, parser_func, output_file, chunk_sizes=[100, 200, 300], overlap=20):
    output_file.write(f"\n📄 Benchmarking: {file_path}\n")

    # Measure parsing time
    start_time = time.time()
    text = parser_func(file_path)
    parse_time = time.time() - start_time
    output_file.write(f"Parsing completed in {parse_time:.2f} seconds\n")

    for size in chunk_sizes:
        start_time = time.time()
        chunks = chunk_text_by_words(text, chunk_size=size, overlap=overlap)
        chunk_time = time.time() - start_time

        avg_words = sum(len(c.split()) for c in chunks) / len(chunks)

        output_file.write(f"\n--- Chunk Size {size} ---\n")
        output_file.write(f"Chunking time: {chunk_time:.2f} seconds\n")
        output_file.write(f"Number of chunks: {len(chunks)}\n")
        output_file.write(f"Average words per chunk: {avg_words:.1f}\n")


# ---------- Run Benchmark ----------
if __name__ == "__main__":
    # Example files (replace with your test documents)
    # pdf_file = "Samples/Aishwarya_Nellutla(2).pdf"
    docx_file = "Samples/DT Batch A1.docx"
    pptx_file = "Samples/sample.pptx"

    # Open output file
    with open("FINAL_benchmarking.txt", "w", encoding="utf-8") as f:
        # benchmark_file(pdf_file, parse_pdf, f)
        benchmark_file(docx_file, parse_docx, f)
        benchmark_file(pptx_file, parse_pptx, f)

    print("✅ Benchmarking completed. Results saved in FINAL_benchmarking.txt")
