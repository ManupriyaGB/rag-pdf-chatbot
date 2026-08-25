import os
import pickle
import hashlib

import faiss
import numpy as np

from utils import load_pdf
from chunker import TextChunker
from embeddings import EmbeddingModel
from retriever import Retriever
from prompt import PromptBuilder
from llm import LLM
from table_loader import TableLoader


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
        self.table_loader = TableLoader()

        self.retriever = None

        self.index_file = os.path.join(
            self.vector_db,
            "index.faiss"
        )

        self.chunk_file = os.path.join(
            self.vector_db,
            "chunks.pkl"
        )

        self.source_hash_file = os.path.join(
            self.vector_db,
            "source_hash.txt"
        )

    # =========================================================
    # FIND ALL FILES
    # =========================================================

    def find_source_files(self):

        source_files = []

        if not os.path.exists(self.data_folder):

            print(f"Data folder not found: {self.data_folder}")

            return source_files

        for root, dirs, files in os.walk(self.data_folder):

            for file in files:

                extension = os.path.splitext(file)[1].lower()

                if extension in [
                    ".pdf",
                    ".csv",
                    ".xlsx",
                    ".xls"
                ]:

                    full_path = os.path.join(root, file)

                    source_files.append(full_path)

        return sorted(source_files)

    # =========================================================
    # CREATE HASH OF SOURCE FILES
    # =========================================================

    def calculate_source_hash(self):

        files = self.find_source_files()

        hasher = hashlib.md5()

        for file_path in files:

            hasher.update(file_path.encode())

            try:

                with open(file_path, "rb") as f:

                    while True:

                        data = f.read(1024 * 1024)

                        if not data:
                            break

                        hasher.update(data)

            except Exception as e:

                print(f"Could not read {file_path}: {e}")

        return hasher.hexdigest()

    # =========================================================
    # CHECK WHETHER DATABASE IS UP TO DATE
    # =========================================================

    def database_is_current(self):

        if not os.path.exists(self.index_file):
            return False

        if not os.path.exists(self.chunk_file):
            return False

        if not os.path.exists(self.source_hash_file):
            return False

        current_hash = self.calculate_source_hash()

        with open(self.source_hash_file, "r") as f:

            saved_hash = f.read().strip()

        return current_hash == saved_hash

    # =========================================================
    # LOAD ALL DOCUMENTS
    # =========================================================

    def load_all_documents(self):

        documents = []

        source_files = self.find_source_files()

        print("=" * 70)
        print("SCANNING DATA FILES")
        print("=" * 70)

        print(f"Files Found : {len(source_files)}")

        if not source_files:

            print("No PDF / CSV / Excel files found!")

            return documents

        for file_path in source_files:

            extension = os.path.splitext(
                file_path
            )[1].lower()

            file_name = os.path.basename(file_path)

            print("\n" + "-" * 60)
            print(f"File : {file_path}")
            print(f"Type : {extension}")

            # =================================================
            # PDF
            # =================================================

            if extension == ".pdf":

                print("Loading PDF...")

                try:

                    text = load_pdf(file_path)

                    chunks = self.chunker.split_text(text)

                    print(
                        f"PDF Chunks Created : {len(chunks)}"
                    )

                    documents.extend(chunks)

                except Exception as e:

                    print(
                        f"ERROR loading PDF {file_name}: {e}"
                    )

            # =================================================
            # CSV / EXCEL
            # =================================================

            elif extension in [
                ".csv",
                ".xlsx",
                ".xls"
            ]:

                print("Loading table...")

                try:

                    rows = self.table_loader.load(
                        file_path
                    )

                    print(
                        f"Table Rows Created : {len(rows)}"
                    )

                    documents.extend(rows)

                except Exception as e:

                    print(
                        f"ERROR loading table "
                        f"{file_name}: {e}"
                    )

        print("\n" + "=" * 70)
        print(
            f"TOTAL SEARCHABLE DOCUMENTS : {len(documents)}"
        )
        print("=" * 70)

        return documents

    # =========================================================
    # BUILD FAISS DATABASE
    # =========================================================

    def build_vector_database(self):

        print("\n")
        print("=" * 70)
        print("BUILDING VECTOR DATABASE")
        print("=" * 70)

        documents = self.load_all_documents()

        if not documents:

            raise ValueError(
                "No documents found. "
                "Put PDF/CSV/XLSX files inside data/."
            )

        print("\nGenerating embeddings...")

        embeddings = (
            self.embedding_model.create_embeddings(
                documents
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        print(
            f"Embedding Dimension : {dimension}"
        )

        print(
            f"Total Vectors : {len(embeddings)}"
        )

        # =====================================================
        # FAISS
        # =====================================================

        print("\nCreating FAISS index...")

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        faiss.write_index(
            index,
            self.index_file
        )

        # =====================================================
        # SAVE DOCUMENTS
        # =====================================================

        with open(
            self.chunk_file,
            "wb"
        ) as f:

            pickle.dump(
                documents,
                f
            )

        # =====================================================
        # SAVE SOURCE HASH
        # =====================================================

        source_hash = self.calculate_source_hash()

        with open(
            self.source_hash_file,
            "w"
        ) as f:

            f.write(source_hash)

        print("\nFAISS database created successfully.")

        print(
            f"Documents saved : {len(documents)}"
        )

    # =========================================================
    # LOAD DATABASE
    # =========================================================

    def load_vector_database(self):

        print("\n")
        print("=" * 70)
        print("CHECKING VECTOR DATABASE")
        print("=" * 70)

        if not self.database_is_current():

            print(
                "Vector database is missing or outdated."
            )

            print(
                "Rebuilding FAISS database..."
            )

            self.build_vector_database()

        else:

            print(
                "Existing Vector Database is up to date."
            )

        # =====================================================
        # RETRIEVER
        # =====================================================

        print("\nInitializing Retriever...")

        self.retriever = Retriever(
            self.embedding_model
        )

        print("Retriever Ready")

    # =========================================================
    # ASK QUESTION
    # =========================================================

    def ask(
        self,
        query,
        chat_history=None
    ):

        if self.retriever is None:

            self.load_vector_database()

        print("\n")
        print("=" * 70)
        print("USER QUESTION")
        print("=" * 70)

        print(query)

        # =====================================================
        # RETRIEVE
        # =====================================================

        print("\nRetrieving relevant information...")

        retrieved_chunks = (
            self.retriever.retrieve(query)
        )

        print("\nRetrieved Chunks:")

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                f"\nChunk {i}:"
            )

            print(chunk)

        # =====================================================
        # BUILD PROMPT
        # =====================================================

        prompt = self.prompt_builder.build_prompt(
            retrieved_chunks,
            query,
            chat_history
        )

        # =====================================================
        # GEMINI
        # =====================================================

        print("\nGenerating answer...")

        answer = self.llm.generate(
            prompt
        )

        print("\nAnswer:")

        print(answer)

        return answer