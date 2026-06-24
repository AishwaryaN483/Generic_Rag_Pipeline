# Generic_Rag_Pipeline
# 📄 RAG-Based Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that enables users to upload documents and ask natural-language questions. The system grounds all answers in the uploaded source material, improving reliability and reducing hallucination compared to vanilla LLM responses.

---

## 🚀 Features

- **Multi-format document support** — Upload PDF, DOCX, and PPTX files
- **Intelligent text chunking** — Splits documents into semantically meaningful segments for precise retrieval
- **Vector similarity search** — Uses FAISS for fast, scalable nearest-neighbor lookup
- **Sentence Transformer embeddings** — Converts text chunks and queries into high-quality dense vectors
- **LLM-powered answers** — LangChain + OpenAI API generates detailed, document-grounded responses
- **Source-faithful responses** — Answers are derived from your uploaded documents, not general model knowledge

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Document Ingestion                   │
│  Upload (PDF / DOCX / PPTX) → Parse → Preprocess       │
│  → Chunk → Embed (Sentence Transformers) → FAISS Index  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Query Pipeline                       │
│  Embed Query → FAISS Similarity Search → Top-k Chunks  │
│  → LangChain Prompt → OpenAI LLM → Final Answer        │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Document Parsing | PyPDF2 / pdfplumber, python-docx, python-pptx |
| Text Embeddings | Sentence Transformers (`sentence-transformers`) |
| Vector Store | FAISS (`faiss-cpu`) |
| LLM Orchestration | LangChain |
| Language Model | OpenAI API (GPT-3.5 / GPT-4) |
| Language | Python 3.9+ |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🔧 Usage

### Ingest a document

```bash
python ingest.py --file path/to/your/document.pdf
```

Supported formats: `.pdf`, `.docx`, `.pptx`

### Ask a question

```bash
python query.py --question "What are the key findings in the report?"
```

### Run the full pipeline interactively

```bash
python main.py
```

---

## 📁 Project Structure

```
rag-document-qa/
│
├── ingestion/
│   ├── parser.py          # PDF, DOCX, PPTX text extraction
│   ├── preprocessor.py    # Text cleaning and normalization
│   └── chunker.py         # Text splitting logic
│
├── embeddings/
│   ├── embedder.py        # Sentence Transformer embedding generation
│   └── vector_store.py    # FAISS index creation and retrieval
│
├── qa/
│   ├── retriever.py       # Query embedding + similarity search
│   └── generator.py       # LangChain + OpenAI answer generation
│
├── main.py                # Entry point for interactive Q&A
├── ingest.py              # CLI for document ingestion
├── query.py               # CLI for querying
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ How It Works

1. **Document Parsing** — Uploaded files are parsed using format-specific libraries to extract raw text while preserving structure.

2. **Preprocessing** — Extracted text is cleaned, normalized, and stripped of irrelevant artifacts.

3. **Chunking** — Text is split into overlapping chunks to ensure context is not lost at chunk boundaries.

4. **Embedding** — Each chunk is converted into a dense vector using a Sentence Transformer model (e.g., `all-MiniLM-L6-v2`).

5. **Indexing** — Chunk vectors are stored in a FAISS index for efficient approximate nearest-neighbor retrieval.

6. **Query Processing** — The user's question is embedded using the same model, then matched against the FAISS index to retrieve the top-k most relevant chunks.

7. **Answer Generation** — Retrieved chunks are passed as context to an OpenAI LLM via a LangChain prompt chain, which generates a detailed, grounded answer.

---

## 📋 Requirements

```
langchain
langchain-openai
openai
sentence-transformers
faiss-cpu
PyPDF2
pdfplumber
python-docx
python-pptx
python-dotenv
numpy
```

---

## 🔑 Configuration

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Number of characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of retrieved chunks per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model for answer generation |

These can be adjusted in `config.py` or via environment variables.

---

## 🧪 Example

```
> python main.py

Upload a document: reports/annual_report_2024.pdf
✅ Document ingested: 142 chunks indexed.

Ask a question: What was the total revenue in Q3?

Answer:
According to the document, the total revenue in Q3 2024 was $4.2 billion,
representing a 12% year-over-year growth driven primarily by the cloud
services segment (Section 3, page 18).
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM orchestration framework
- [Sentence Transformers](https://www.sbert.net/) for the embedding models
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI for vector similarity search
- [OpenAI](https://openai.com/) for the language model API
