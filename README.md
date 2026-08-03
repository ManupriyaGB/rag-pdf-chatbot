# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) based PDF Question Answering system built using Python, Hugging Face Transformers, FAISS, Ollama, and Streamlit.

The application allows users to upload a PDF, create a vector database from its content, and ask questions based only on the uploaded document.

---

# Features

- Upload any PDF document
- Extract text from PDF
- Split document into chunks
- Generate embeddings using Hugging Face
- Store embeddings using FAISS
- Retrieve relevant chunks
- Generate context-aware answers using Ollama
- Simple Streamlit interface
- Modular project structure

---

# Project Structure

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
│   └── test_rag.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# RAG Pipeline

```
                    OFFLINE

              PDF Document
                    │
                    ▼
             Extract Text
                    │
                    ▼
              Text Chunking
                    │
                    ▼
          Generate Embeddings
                    │
                    ▼
             FAISS Vector DB
                    │
                    ▼
         vector_db/index.faiss
                    │
                    ▼
         vector_db/chunks.pkl


=====================================================


                    ONLINE

              User Question
                    │
                    ▼
        Query Embedding Creation
                    │
                    ▼
            Similarity Search
                    │
                    ▼
          Top-K Relevant Chunks
                    │
                    ▼
             Prompt Creation
                    │
                    ▼
             Ollama LLM
                    │
                    ▼
               Final Answer
```

---

# Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- FAISS
- Ollama
- PyTorch
- NumPy
- PyPDF

---

# Installation

Clone the repository

```bash
git clone https://github.com/ManupriyaGB/rag-pdf-chatbot.git
```

Move into the project directory

```bash
cd rag-pdf-chatbot
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download and install Ollama.

Pull the model

```bash
ollama pull llama3.2
```

Verify installation

```bash
ollama list
```

---

# Run the Application

```bash
streamlit run app/main.py
```

---

# Test the Pipeline

```bash
python tests/test_rag.py
```

---

# Example Workflow

1. Upload a PDF document.
2. Click **Build Knowledge Base**.
3. The application extracts text from the PDF.
4. The document is split into chunks.
5. Embeddings are generated.
6. Embeddings are stored in FAISS.
7. Ask any question related to the uploaded PDF.
8. The system retrieves the most relevant chunks.
9. Ollama generates the final answer using the retrieved context.

---

# Vector Database

The `vector_db` directory stores:

```
vector_db/
│
├── index.faiss
└── chunks.pkl
```

- **index.faiss** → Stores embedding vectors.
- **chunks.pkl** → Stores the original text chunks.

---

# Screenshots

```
screenshots/
│
├── home.png
├── upload_pdf.png
├── building_vector_db.png
├── asking_question.png
└── final_answer.png
```

Add screenshots after running the application.

---

# Future Improvements

- Support multiple PDFs
- ChromaDB integration
- Pinecone integration
- Qdrant integration
- Hybrid Search
- Conversation Memory
- Chat History
- GPU acceleration
- Docker support
- Cloud deployment


