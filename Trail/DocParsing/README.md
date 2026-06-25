# DOCX Parser (Text, Tables & Images)

This script extracts text, tables, and images from `.docx` files and saves the extracted images in a folder placed in the same directory as the script.


## Features
- Extracts all text from paragraphs (ignores blank lines).
- Extracts all tables with row and cell data.
- Extracts and saves images from the DOCX file into an `extracted_images/` folder.
- Prints parsing time for benchmarking.



## 🛠 Requirements

Install the required library:

```bash
pip install python-docx
