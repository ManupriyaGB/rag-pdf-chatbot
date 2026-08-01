# 📚 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents using Large Language Models (LLMs).

The project demonstrates the complete RAG pipeline including document loading, text chunking, embedding generation, vector database indexing, semantic retrieval, prompt engineering, and answer generation.

---

## Features

- Load PDF documents
- Extract text from PDFs
- Split documents into chunks
- Generate text embeddings
- Store embeddings in a FAISS vector database
- Retrieve relevant document chunks
- Generate answers using an LLM
- Streamlit-based user interface
- Modular project architecture

---

## Project Structure

```
rag-pdf-chatbot/
│
├── app/
│   ├── main.py
│   ├── utils.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── rag.py
│   ├── retriever.py
│   ├── prompt.py
│   └── llm.py
│
├── data/
│
├── vector_db/
│
├── notebooks/
│
├── screenshots/
│
├── tests/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# RAG Architecture

## Offline Pipeline

```
PDF
 │
 ▼
Document Loader
 │
 ▼
Text Chunking
 │
 ▼
Embedding Generation
 │
 ▼
FAISS Vector Database
```

---

## Online Pipeline

```
User Question
      │
      ▼
Question Embedding
      │
      ▼
Similarity Search
      │
      ▼
Relevant Chunks
      │
      ▼
Prompt Builder
      │
      ▼
LLM
      │
      ▼
Final Answer
```

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Sentence Transformers
- FAISS
- Ollama
- PyTorch
- Hugging Face Transformers

---

## Installation

Clone the repository:

```bash
git clone git@github.com:ManupriyaGB/rag-pdf-chatbot.git
```

Move into the project:

```bash
cd rag-pdf-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/main.py
```

---

## Future Improvements

- Support multiple PDFs
- Hybrid Search
- Metadata Filtering
- Chat History
- Source Citation
- OCR Support
- Cloud Deployment
- Multi-LLM Support



