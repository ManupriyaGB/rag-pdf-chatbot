# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) based PDF Question Answering system built using Python, Hugging Face Transformers, FAISS, Ollama, and Streamlit.

The application Automatically reads all PDF files from the data/ folder, create a vector database from its content, and ask questions based only on the uploaded document.

---

# Features

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

         PDFs from data/
                │
                ▼
         Extract Text
                │
                ▼
           Chunking
                │
                ▼
     Generate Embeddings
                │
                ▼
      Create FAISS Index
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
# Before Running

Copy one or more PDF files into the `data/` folder.

Example

```text
data/
│
├── attention_is_all_you_need.pdf
├── bert.pdf
└── llama.pdf
```

If the vector database does not exist, it will be created automatically when the application starts.
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

1. Place one or more PDF files inside the data/ folder.
2. Start the application.
3. The application automatically builds the vector database if it does not exist.
4. Enter your question.
5. Relevant document chunks are retrieved.
6. Ollama generates the final answer.

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

---

# Future Improvements

- ChromaDB integration
- Pinecone integration
- Qdrant integration
- Hybrid Search
- Conversation Memory
- Chat History
- GPU acceleration
- Docker support
- Cloud deployment


