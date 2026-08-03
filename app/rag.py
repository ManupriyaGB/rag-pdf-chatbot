import os
import faiss
import pickle
import numpy as np

from utils import load_pdf
from chunker import TextChunker
from embeddings import EmbeddingModel
from retriever import Retriever
from prompt import PromptBuilder
from llm import LLM


class RAGPipeline:

    def __init__(self):

        print("=" * 70)
        print("Initializing RAG Pipeline")
        print("=" * 70)

        self.chunker = TextChunker()

        self.embedding_model = EmbeddingModel()

        self.prompt_builder = PromptBuilder()

        self.llm = LLM()

        self.retriever = None

    # ====================================================
    # OFFLINE PIPELINE
    # ====================================================

    def build_vector_database(self, pdf_path):

        print("\nLoading PDF...")

        text = load_pdf(pdf_path)

        print("PDF Loaded")

        print("\nChunking...")

        chunks = self.chunker.split_text(text)

        print(f"Total Chunks : {len(chunks)}")

        print("\nGenerating Embeddings...")

        embeddings = self.embedding_model.create_embeddings(chunks)

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        print("Embedding Shape :", embeddings.shape)

        dimension = embeddings.shape[1]

        print("\nCreating FAISS Index...")

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        os.makedirs("vector_db", exist_ok=True)

        faiss.write_index(
            index,
            "vector_db/index.faiss"
        )

        with open(
            "vector_db/chunks.pkl",
            "wb"
        ) as file:

            pickle.dump(chunks, file)

        print("\nKnowledge Base Ready")

    # ====================================================
    # LOAD DATABASE
    # ====================================================

    def load_database(self):

        self.retriever = Retriever(
            self.embedding_model
        )

    # ====================================================
    # ONLINE PIPELINE
    # ====================================================

    def ask(self, query):

        print("\nStarting Online Pipeline")

        retrieved_chunks = self.retriever.retrieve(
            query
        )

        prompt = self.prompt_builder.build_prompt(
            retrieved_chunks,
            query
        )

        answer = self.llm.generate(
            prompt
        )

        return answer