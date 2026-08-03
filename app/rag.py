import os
import pickle
import faiss
import numpy as np

from app.utils import load_pdf
from app.chunker import TextChunker
from app.embeddings import EmbeddingModel
from app.retriever import Retriever
from app.prompt import PromptBuilder
from app.llm import LLM


class RAGPipeline:

    def __init__(self):

        print("=" * 70)
        print("Initializing RAG Pipeline")
        print("=" * 70)

        self.data_folder = "data"
        self.vector_db = "vector_db"

        os.makedirs(self.vector_db, exist_ok=True)

        self.chunker = TextChunker()

        self.embedding_model = EmbeddingModel()

        self.prompt_builder = PromptBuilder()

        self.llm = LLM()

        self.retriever = None

        self.index_file = os.path.join(
            self.vector_db,
            "index.faiss"
        )

        self.chunk_file = os.path.join(
            self.vector_db,
            "chunks.pkl"
        )

    # ==========================================================
    # Read all PDFs
    # ==========================================================

    def load_all_pdfs(self):

        print("\nSearching PDFs...")

        pdf_files = [
            file
            for file in os.listdir(self.data_folder)
            if file.lower().endswith(".pdf")
        ]

        if len(pdf_files) == 0:

            raise FileNotFoundError(
                "No PDF files found inside data folder."
            )

        print(f"Found {len(pdf_files)} PDF(s)\n")

        all_chunks = []

        for pdf in pdf_files:

            pdf_path = os.path.join(
                self.data_folder,
                pdf
            )

            print("-" * 60)
            print(f"Reading : {pdf}")

            text = load_pdf(pdf_path)

            chunks = self.chunker.split_text(text)

            print(f"Chunks Created : {len(chunks)}")

            all_chunks.extend(chunks)

        print("\nTotal Chunks :", len(all_chunks))

        return all_chunks

    # ==========================================================
    # Build Vector Database
    # ==========================================================

    def build_vector_database(self):

        print("\n" + "=" * 70)
        print("BUILDING VECTOR DATABASE")
        print("=" * 70)

        chunks = self.load_all_pdfs()

        print("\nGenerating Embeddings...\n")

        embeddings = self.embedding_model.create_embeddings(chunks)

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        print("\nEmbedding Shape :", embeddings.shape)

        dimension = embeddings.shape[1]

        print("\nCreating FAISS Index...")

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        print(f"\nTotal Vectors Stored : {index.ntotal}")

        print("\nSaving FAISS Index...")

        faiss.write_index(
            index,
            self.index_file
        )

        print("Index Saved Successfully")

        print("\nSaving Chunks...")

        with open(
            self.chunk_file,
            "wb"
        ) as file:

            pickle.dump(
                chunks,
                file
            )

        print("Chunks Saved Successfully")

        print("\nKnowledge Base Created Successfully")

    # ==========================================================
    # Load Existing Vector Database
    # ==========================================================

    def load_vector_database(self):

        print("\n" + "=" * 70)
        print("LOADING VECTOR DATABASE")
        print("=" * 70)

        if not os.path.exists(self.index_file):

            print("\nNo Existing Vector Database Found.")

            print("Creating New Database...\n")

            self.build_vector_database()

        else:

            print("\nExisting Vector Database Found.")

        self.retriever = Retriever(
            self.embedding_model
        )

        print("\nRetriever Ready")
        # ==========================================================
    # Ask Question
    # ==========================================================

    def ask(self, query):

        print("\n" + "=" * 70)
        print("ONLINE RAG PIPELINE")
        print("=" * 70)

        if self.retriever is None:

            self.load_vector_database()

        print("\nUser Question:")
        print(query)

        print("\nRetrieving Relevant Chunks...\n")

        retrieved_chunks = self.retriever.retrieve(query)

        print("\nRetrieved Chunks :")

        for i, chunk in enumerate(retrieved_chunks, start=1):

            print("-" * 60)
            print(f"Chunk {i}")
            print(chunk[:300])
            print()

        print("=" * 70)
        print("BUILDING PROMPT")
        print("=" * 70)

        prompt = self.prompt_builder.build_prompt(
            retrieved_chunks,
            query
        )

        print("\nPrompt Created Successfully")

        print("\n" + "=" * 70)
        print("GENERATING ANSWER")
        print("=" * 70)

        answer = self.llm.generate(prompt)

        print("\nAnswer Generated Successfully\n")

        return answer
