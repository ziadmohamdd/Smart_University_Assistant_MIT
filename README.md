# 🎓 Smart University Assistant

A Retrieval-Augmented Generation (RAG) based university assistant that answers questions using information retrieved from a curated document collection. The system combines semantic search with a locally hosted Large Language Model (LLM) to generate grounded answers with cited sources.

## ✨ Features

* 🔎 **Semantic Document Retrieval** using Sentence Transformers
* 🧠 **Retrieval-Augmented Generation (RAG)** for grounded answers
* 🗃️ **ChromaDB Vector Database** for persistent document embeddings
* 🤖 **Local LLM Generation** using Ollama
* ⚡ **FastAPI Backend** for serving the RAG pipeline
* 💬 **Streamlit Chat Interface** for user interaction
* 📚 **Source Citations** displayed with generated answers
* ❤️ **Health Monitoring** for retrieval and generation services
* 🧪 **Automated API Testing** with Pytest
* 📓 Jupyter notebooks for data collection, processing, embedding, and evaluation

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │   University User   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Frontend  │
                 │    Chat Interface   │
                 └──────────┬──────────┘
                            │ HTTP
                            ▼
                 ┌─────────────────────┐
                 │    FastAPI API      │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
        ┌──────────────────┐   ┌──────────────────┐
        │ Retrieval Service│   │Generation Service│
        └────────┬─────────┘   └────────┬─────────┘
                 │                      │
                 ▼                      ▼
        ┌──────────────────┐   ┌──────────────────┐
        │ Sentence        │   │     Ollama       │
        │ Transformers    │   │   Local LLM       │
        └────────┬─────────┘   └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │    ChromaDB      │
        │   Vector Store   │
        └──────────────────┘
```

## 🔄 RAG Pipeline

The project follows a complete RAG workflow:

### 1. Data Collection

University-related documents are collected and standardized into structured JSON files containing:

* Document metadata
* Titles
* Pages
* Headings
* Paragraphs
* Source URLs
* Document types
* Language information

The data collection process is documented in:

```text
notebooks/collecting_data.ipynb
```

### 2. Document Processing

Documents are cleaned and transformed into a standardized structure before entering the RAG pipeline.

```text
notebooks/document_processing.ipynb
```

### 3. Section-Aware Chunking

Instead of simply splitting documents by page, the system identifies document sections using headings.

Small neighboring sections are merged before applying the sliding-window chunking strategy.

Configuration:

```text
Chunk size:       220 words
Chunk overlap:     40 words
Section threshold: 60 words
```

This approach helps preserve the relationship between headings and their corresponding content while reducing unrelated information inside individual chunks.

### 4. Embedding Generation

Document chunks are converted into semantic embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings allow the system to find documents based on semantic meaning rather than simple keyword matching.

### 5. Vector Database

The generated embeddings and document metadata are stored in a persistent **ChromaDB** collection:

```text
rag_assistant_chunks
```

The backend loads this existing vector store rather than rebuilding it for every application startup.

### 6. Retrieval

When a user asks a question:

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB Semantic Search
      ↓
Top 5 Relevant Chunks
      ↓
Context Construction
```

The default retrieval value is:

```text
RETRIEVAL_TOP_K = 5
```

### 7. Answer Generation

The retrieved context is passed to a locally running Ollama model.

Default model:

```text
llama3.2:3b
```

The generation configuration uses:

```text
Temperature: 0.0
Context window: 8192
Maximum prediction: 400 tokens
```

The system is designed to keep answers grounded in the retrieved context and extract the sources cited by the generated answer.

## 🛠️ Technology Stack

| Component         | Technology             |
| ----------------- | ---------------------- |
| Frontend          | Streamlit              |
| Backend           | FastAPI                |
| Language          | Python                 |
| Vector Database   | ChromaDB               |
| Embeddings        | Sentence Transformers  |
| LLM Runtime       | Ollama                 |
| Default LLM       | Llama 3.2 3B           |
| API Communication | REST / HTTP            |
| Testing           | Pytest                 |
| Data Processing   | Pandas / Python        |
| Development       | Jupyter / Google Colab |

## 📁 Project Structure

```text
SmartUniversityAssistant/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       └── query.py
│   │   │
│   │   ├── core/
│   │   │   └── config.py
│   │   │
│   │   ├── schemas/
│   │   │   └── query.py
│   │   │
│   │   ├── services/
│   │   │   ├── retrieval.py
│   │   │   └── generation.py
│   │   │
│   │   ├── utils/
│   │   │   └── logging_config.py
│   │   │
│   │   └── main.py
│   │
│   ├── tests/
│   │   └── test_query.py
│   │
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── requirements.txt
│   └── .env.example
│
└── notebooks/
    ├── collecting_data.ipynb
    ├── document_processing.ipynb
    └── rag_pipeline_final.ipynb
```

## 🚀 Installation

### Prerequisites

Make sure you have installed:

* Python 3.10+
* Ollama
* Git

You also need the persisted ChromaDB vector store generated by the RAG notebook.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartUniversityAssistant.git
cd SmartUniversityAssistant
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure the Backend

Copy:

```text
.env.example
```

to:

```text
.env
```

The default configuration expects:

```env
CHROMA_PERSIST_DIR=./data/vector_store
CHROMA_COLLECTION_NAME=rag_assistant_chunks
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_TOP_K=5
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0.0
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=400
```

### 4. Install and Prepare Ollama

Install Ollama and pull the required model:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama service is running before sending queries.

### 5. Add the Vector Store

The FastAPI backend expects the persisted ChromaDB database at:

```text
backend/data/vector_store/
```

The vector store is generated by:

```text
notebooks/rag_pipeline_final.ipynb
```

The notebook performs the embedding and ChromaDB persistence steps.

> **Note:** The vector database may not be included in the Git repository because of its size. Generate it using the notebook or place the provided persisted vector store in the expected directory.

## ▶️ Running the Application

### Start the Backend

From the `backend` directory:

```bash
uvicorn app.main:app --reload
```

The API will run at:

```text
http://localhost:8000
```

### Backend Health Check

Open:

```text
http://localhost:8000/health
```

The health endpoint reports whether:

* The retrieval service is ready
* The generation service is ready

### Start the Frontend

Open a second terminal:

```bash
cd frontend
pip install -r requirements.txt
```

Create `.env` from `.env.example` and configure:

```env
API_BASE_URL=http://localhost:8000
```

Then run:

```bash
streamlit run app.py
```

The Streamlit interface will open in your browser.

## 🔌 API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "retrieval_ready": true,
  "generation_ready": true
}
```

### Ask a Question

```http
POST /query
```

Request:

```json
{
  "question": "What information is available about the university?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    "..."
  ]
}
```

## 🧪 Testing

The backend includes API tests using Pytest.

Run:

```bash
cd backend
pytest
```

The tests cover API behavior such as valid queries, invalid requests, and service availability.

## 📊 Evaluation

The RAG notebook also contains a retrieval and generation evaluation workflow.

The evaluation checks:

* Retrieval relevance
* Answer grounding
* Citation presence
* Expected answer information
* Out-of-domain question handling
* Generation/system errors
* Failure cases and possible mitigations

This helps distinguish retrieval problems from generation failures and unsupported questions.

## 🔐 Environment Variables

Sensitive or machine-specific configuration should be stored in `.env` files.

Example:

```text
backend/.env.example
frontend/.env.example
```

Do **not** commit real `.env` files containing private credentials or machine-specific secrets.

## 🎯 Project Goals

The project demonstrates how a production-oriented RAG application can combine:

```text
Document Processing
        ↓
Semantic Embeddings
        ↓
Vector Database
        ↓
Information Retrieval
        ↓
Context-Aware Prompting
        ↓
Local LLM Generation
        ↓
Source-Cited Answer
```

The architecture separates the data-processing pipeline from the serving layer, allowing the FastAPI backend to reuse a pre-built vector store without rebuilding embeddings for every request.

## 👨‍💻 Author

**Ziad Elsayed & Omar Sayed**

AI & Data Science Student
Computer Science and Information Technology

---

⭐ If you find this project useful, consider giving the repository a star!
