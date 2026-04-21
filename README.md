
title: Classwork-gen
app_file: src/app.py
sdk: gradio
sdk_version: 6.12.0


# ClassworkAi
https://huggingface.co/spaces/its-Brayan/Classwork-gen
A RAG (Retrieval Augmented Generation) application that answers questions based on PDF documents using semantic search and AI.

## Features

- **PDF Document Loading** — Loads and processes PDF files from a data directory
- **Semantic Search** — Uses vector embeddings to find relevant content
- **AI-Powered Answers** — Leverages Google Gemini to generate responses based on retrieved context
- **Persistent Vector Store** — ChromaDB stores embeddings locally for fast retrieval

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDFs      │────>│  VectorDB    │────>│   RAG       │
│ (data/)     │     │ (ChromaDB)   │     │  Pipeline   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                    ┌──────┴──────┐       ┌────┴────┐
                    │ Sentence    │       │ Gemini  │
                    │ Transformer │       │   LLM   │
                    └─────────────┘       └─────────┘
```

## Project Structure

```
ClassworkAi/
├── config/
│   ├── config.yaml          # LLM and embedding model settings
│   └── prompt_config.yaml   # QA prompt templates
├── data/                    # PDF documents go here
├── src/
│   ├── app.py               # Main application entry point
│   ├── vectordb.py          # Vector database wrapper
│   └── config_loader.py     # Configuration loading utilities
├── chroma_db/               # Persisted vector database (created at runtime)
├── requirements.txt         # Python dependencies
└── .env                     # API keys (create your own)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
COLLECTION_NAME=classwork_collection
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Add Documents

Place PDF files in the `data/` directory.

### 4. Run

```bash
cd src
python app.py
```

## Usage

1. The app loads all PDFs from the `data/` directory
2. Chunks the text and creates embeddings
3. Enter questions when prompted — type `quit` to exit

Example:
```
Enter a question (or 'quit' to exit): What is the main topic of the documents?
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `llm` | LLM model name | `gemini-3-flash-preview` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `COLLECTION_NAME` | ChromaDB collection name | `classwork_collection` |

## Dependencies

- **chromadb** — Vector database
- **sentence-transformers** — Text embeddings
- **langchain-google-genai** — Google Gemini integration
- **langchain** — RAG pipeline components
- **PyMuPDF** — PDF loading

