## PPTX Parser 

This project compares different techniques for extracting text from PowerPoint (.pptx) files and measures their performance. It allows you to evaluate which method works best depending on speed, accuracy, and OCR requirements.

## 🚀 Features

Extracts text from PowerPoint files using multiple techniques:

Direct Parsing (python-pptx)
Regex Cleanup
Shape/TextFrame Iteration
OCR-based extraction (pytesseract + Pillow)
Benchmarks each method using time.perf_counter()
Reports execution time comparison across techniques
Supports .pptx files with both editable text and images with text
