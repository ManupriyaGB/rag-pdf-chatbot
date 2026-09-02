import os
import pickle
import hashlib
import re

import faiss
import numpy as np
import pandas as pd

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

        # =====================================================
        # PATHS
        # =====================================================

        self.data_folder = "data"
        self.vector_db = "vector_db"

        os.makedirs(
            self.vector_db,
            exist_ok=True
        )

        # =====================================================
        # COMPONENTS
        # =====================================================

        self.chunker = TextChunker()

        self.embedding_model = EmbeddingModel()

        self.prompt_builder = PromptBuilder()

        self.llm = LLM()

        self.table_loader = TableLoader()

        # IMPORTANT:
        # Retriever needs embedding_model
        self.retriever = None

        # =====================================================
        # VECTOR DATABASE FILES
        # =====================================================

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
    # FIND SOURCE FILES
    # =========================================================

    def find_source_files(self):

        source_files = []

        if not os.path.exists(
            self.data_folder
        ):

            print(
                f"Data folder not found: "
                f"{self.data_folder}"
            )

            return source_files

        for root, dirs, files in os.walk(
            self.data_folder
        ):

            for file in files:

                extension = os.path.splitext(
                    file
                )[1].lower()

                if extension in [
                    ".pdf",
                    ".csv",
                    ".xlsx",
                    ".xls"
                ]:

                    full_path = os.path.join(
                        root,
                        file
                    )

                    source_files.append(
                        full_path
                    )

        return sorted(
            source_files
        )

    # =========================================================
    # SOURCE HASH
    # =========================================================

    def calculate_source_hash(self):

        files = self.find_source_files()

        hasher = hashlib.md5()

        for file_path in files:

            hasher.update(
                file_path.encode()
            )

            try:

                with open(
                    file_path,
                    "rb"
                ) as f:

                    while True:

                        data = f.read(
                            1024 * 1024
                        )

                        if not data:
                            break

                        hasher.update(
                            data
                        )

            except Exception as e:

                print(
                    f"Could not read "
                    f"{file_path}: {e}"
                )

        return hasher.hexdigest()

    # =========================================================
    # DATABASE CURRENT?
    # =========================================================

    def database_is_current(self):

        if not os.path.exists(
            self.index_file
        ):
            return False

        if not os.path.exists(
            self.chunk_file
        ):
            return False

        if not os.path.exists(
            self.source_hash_file
        ):
            return False

        current_hash = (
            self.calculate_source_hash()
        )

        with open(
            self.source_hash_file,
            "r"
        ) as f:

            saved_hash = f.read().strip()

        return (
            current_hash == saved_hash
        )

    # =========================================================
    # LOAD ALL DOCUMENTS
    # =========================================================

    def load_all_documents(self):

        documents = []

        source_files = (
            self.find_source_files()
        )

        print("=" * 70)
        print("SCANNING DATA FILES")
        print("=" * 70)

        print(
            f"Files Found : "
            f"{len(source_files)}"
        )

        if not source_files:

            print(
                "No PDF / CSV / Excel files found!"
            )

            return documents

        for file_path in source_files:

            extension = os.path.splitext(
                file_path
            )[1].lower()

            file_name = os.path.basename(
                file_path
            )

            print("\n" + "-" * 60)

            print(
                f"File : {file_path}"
            )

            print(
                f"Type : {extension}"
            )

            # =================================================
            # PDF
            # =================================================

            if extension == ".pdf":

                print(
                    "Loading PDF..."
                )

                try:

                    text = load_pdf(
                        file_path
                    )

                    chunks = (
                        self.chunker.split_text(
                            text
                        )
                    )

                    print(
                        f"PDF Chunks Created : "
                        f"{len(chunks)}"
                    )

                    documents.extend(
                        chunks
                    )

                except Exception as e:

                    print(
                        f"ERROR loading PDF "
                        f"{file_name}: {e}"
                    )

            # =================================================
            # CSV / EXCEL
            # =================================================

            elif extension in [
                ".csv",
                ".xlsx",
                ".xls"
            ]:

                print(
                    "Loading table..."
                )

                try:

                    rows = (
                        self.table_loader.load(
                            file_path
                        )
                    )

                    print(
                        f"Table Rows Created : "
                        f"{len(rows)}"
                    )

                    documents.extend(
                        rows
                    )

                except Exception as e:

                    print(
                        f"ERROR loading table "
                        f"{file_name}: {e}"
                    )

        print("\n" + "=" * 70)

        print(
            f"TOTAL SEARCHABLE DOCUMENTS : "
            f"{len(documents)}"
        )

        print("=" * 70)

        return documents

    # =========================================================
    # BUILD VECTOR DATABASE
    # =========================================================

    def build_vector_database(self):

        print("\n")
        print("=" * 70)
        print("BUILDING VECTOR DATABASE")
        print("=" * 70)

        documents = (
            self.load_all_documents()
        )

        if not documents:

            raise ValueError(
                "No documents found. "
                "Put PDF/CSV/XLSX files inside data/."
            )

        print(
            "\nGenerating embeddings..."
        )

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
            f"Embedding Dimension : "
            f"{dimension}"
        )

        print(
            f"Total Vectors : "
            f"{len(embeddings)}"
        )

        # =====================================================
        # FAISS
        # =====================================================

        print(
            "\nCreating FAISS index..."
        )

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(
            embeddings
        )

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
        # SOURCE HASH
        # =====================================================

        source_hash = (
            self.calculate_source_hash()
        )

        with open(
            self.source_hash_file,
            "w"
        ) as f:

            f.write(
                source_hash
            )

        print(
            "\nFAISS database created successfully."
        )

        print(
            f"Documents saved : "
            f"{len(documents)}"
        )

    # =========================================================
    # LOAD VECTOR DATABASE
    # =========================================================

    def load_vector_database(self):

        print("\n")
        print("=" * 70)
        print("CHECKING VECTOR DATABASE")
        print("=" * 70)

        if not self.database_is_current():

            print(
                "Vector database is missing "
                "or outdated."
            )

            print(
                "Rebuilding FAISS database..."
            )

            self.build_vector_database()

        else:

            print(
                "Existing Vector Database "
                "is up to date."
            )

        # =====================================================
        # RETRIEVER
        # =====================================================

        print(
            "\nInitializing Retriever..."
        )

        # IMPORTANT FIX
        self.retriever = Retriever(
            self.embedding_model
        )

        print(
            "Retriever Ready"
        )

    # =========================================================
    # LOAD TABLE DIRECTLY
    #
    # This is NOT SQL.
    #
    # We use pandas only to read the table so that
    # we don't lose rows because of FAISS top-k retrieval.
    # =========================================================

    def load_tables(self):

        tables = []

        source_files = (
            self.find_source_files()
        )

        for file_path in source_files:

            extension = os.path.splitext(
                file_path
            )[1].lower()

            if extension not in [
                ".csv",
                ".xlsx",
                ".xls"
            ]:

                continue

            try:

                if extension == ".csv":

                    df = pd.read_csv(
                        file_path
                    )

                else:

                    df = pd.read_excel(
                        file_path
                    )

                # Remove completely empty rows
                df = df.dropna(
                    how="all"
                )

                # Normalize column names
                df.columns = [
                    str(col).strip()
                    for col in df.columns
                ]

                tables.append(
                    {
                        "file": file_path,
                        "data": df
                    }
                )

                print(
                    f"Table loaded directly: "
                    f"{file_path}"
                )

                print(
                    f"Rows    : {len(df)}"
                )

                print(
                    f"Columns : "
                    f"{list(df.columns)}"
                )

            except Exception as e:

                print(
                    f"Could not load table "
                    f"{file_path}: {e}"
                )

        return tables

    # =========================================================
    # CONVERT DATAFRAME TO TEXT
    # =========================================================

    def dataframe_to_text(
        self,
        df,
        source
    ):

        lines = []

        lines.append(
            f"Source file: {source}"
        )

        lines.append(
            "Columns: "
            + ", ".join(
                str(c)
                for c in df.columns
            )
        )

        lines.append("")

        for index, row in df.iterrows():

            values = []

            for column in df.columns:

                value = row[column]

                if pd.isna(value):
                    value = ""

                values.append(
                    f"{column}: {value}"
                )

            lines.append(
                " | ".join(values)
            )

        return "\n".join(
            lines
        )

    # =========================================================
    # NORMALIZE QUERY
    # =========================================================

    def normalize_text(
        self,
        text
    ):

        text = str(
            text
        ).lower()

        text = re.sub(
            r"[^a-z0-9\s.]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =========================================================
    # CHECK WHETHER QUESTION IS TABLE RELATED
    #
    # This used to rely ONLY on a fixed list of English words
    # ("employee", "salary", "department", ...). That silently
    # broke for any uploaded table outside that one domain --
    # e.g. a question like "Lakimsetti Mohan project name" or
    # "Total vs Billable hours for May-Jul" matched none of
    # those words, so it never even tried the table search path.
    #
    # Now, in ADDITION to the static hint words (which still
    # catch generic phrasing like "who" / "list"), we check
    # whether the question mentions any actual COLUMN NAME from
    # whatever tables are currently loaded. That makes table
    # detection work automatically for any uploaded CSV/Excel,
    # regardless of its subject matter.
    # =========================================================

    def is_table_question(
        self,
        query,
        tables=None
    ):

        q = self.normalize_text(
            query
        )

        table_keywords = [

            "employee",
            "employees",

            "salary",
            "salaries",

            "department",
            "departments",

            "performance",
            "rating",

            "city",
            "location",

            "working",
            "works",

            "people",
            "person",

            "who",
            "list",

            "below",
            "above",

            "greater",
            "less",

            "higher",
            "lower",

            "between",

            "average",
            "maximum",
            "minimum",

            "highest",
            "lowest",

            "details",

            "records",

            "data",

            "table"
        ]

        if any(
            keyword in q
            for keyword in table_keywords
        ):
            return True

        # ---------------------------------------------------
        # DYNAMIC CHECK: does the question mention a column
        # name from any currently loaded table?
        # ---------------------------------------------------

        if tables is None:
            tables = self.load_tables()

        if not tables:
            return False

        query_words = set(
            word
            for word in q.split()
            if len(word) >= 3
        )

        if not query_words:
            return False

        for table in tables:

            for column in table["data"].columns:

                column_words = set(
                    word
                    for word in self.normalize_text(
                        column
                    ).split()
                    if len(word) >= 3
                )

                if query_words & column_words:
                    return True

        return False

    # =========================================================
    # FIND NUMERIC CONDITION
    # =========================================================

    def get_numeric_condition(
        self,
        query
    ):

        q = self.normalize_text(
            query
        )

        # below / less than / under
        match = re.search(
            r"(below|less than|under|lower than)\s+(\d+(?:\.\d+)?)",
            q
        )

        if match:

            return (
                "lt",
                float(
                    match.group(2)
                )
            )

        # above / greater than / over
        match = re.search(
            r"(above|greater than|over|higher than)\s+(\d+(?:\.\d+)?)",
            q
        )

        if match:

            return (
                "gt",
                float(
                    match.group(2)
                )
            )

        # equal
        match = re.search(
            r"(equal to|equals|rating of)\s+(\d+(?:\.\d+)?)",
            q
        )

        if match:

            return (
                "eq",
                float(
                    match.group(2)
                )
            )

        return None

    # =========================================================
    # TABLE RETRIEVAL
    # =========================================================

    def retrieve_from_tables(
        self,
        query,
        tables=None
    ):

        if tables is None:
            tables = self.load_tables()

        if not tables:

            return []

        q = self.normalize_text(
            query
        )

        results = []

        numeric_condition = (
            self.get_numeric_condition(
                query
            )
        )

        for table in tables:

            df = table["data"]

            source = table["file"]

            # =================================================
            # NUMERIC FILTER
            # =================================================

            if numeric_condition:

                condition, value = (
                    numeric_condition
                )

                numeric_columns = []

                for column in df.columns:

                    converted = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    )

                    if converted.notna().sum() > 0:

                        numeric_columns.append(
                            (
                                column,
                                converted
                            )
                        )

                matching_rows = []

                # Prefer columns related to rating,
                # salary, score, age etc.
                preferred_columns = []

                for column, converted in numeric_columns:

                    column_name = (
                        self.normalize_text(
                            column
                        )
                    )

                    if any(
                        word in column_name
                        for word in [
                            "rating",
                            "performance",
                            "score",
                            "salary",
                            "age",
                            "marks"
                        ]
                    ):

                        preferred_columns.append(
                            (
                                column,
                                converted
                            )
                        )

                search_columns = (
                    preferred_columns
                    if preferred_columns
                    else numeric_columns
                )

                for index, row in df.iterrows():

                    matched = False

                    for column, converted in search_columns:

                        number = converted.loc[
                            index
                        ]

                        if pd.isna(number):
                            continue

                        number = float(
                            number
                        )

                        if condition == "lt":
                            matched = (
                                number < value
                            )

                        elif condition == "gt":
                            matched = (
                                number > value
                            )

                        elif condition == "eq":
                            matched = (
                                abs(
                                    number - value
                                ) < 0.000001
                            )

                        if matched:
                            break

                    if matched:

                        matching_rows.append(
                            row
                        )

                if matching_rows:

                    result_df = pd.DataFrame(
                        matching_rows
                    )

                    results.append(
                        self.dataframe_to_text(
                            result_df,
                            source
                        )
                    )

                continue

            # =================================================
            # TEXT MATCHING
            # =================================================

            matched_indices = []

            # Convert every row into searchable text
            for index, row in df.iterrows():

                row_text = " ".join(
                    str(value)
                    for value in row.values
                    if not pd.isna(value)
                )

                row_text_normalized = (
                    self.normalize_text(
                        row_text
                    )
                )

                # Extract useful query words
                query_words = [
                    word
                    for word in q.split()
                    if len(word) >= 3
                ]

                score = 0

                for word in query_words:

                    if word in row_text_normalized:

                        score += 1

                if score > 0:

                    matched_indices.append(
                        (
                            index,
                            score
                        )
                    )

            # =================================================
            # IF QUESTION IS BROAD
            # RETURN COMPLETE TABLE
            # =================================================

            broad_question = any(
                phrase in q
                for phrase in [
                    "who all",
                    "list all",
                    "show all",
                    "all employees",
                    "all people",
                    "complete details",
<<<<<<< HEAD
                    "all details"
                ]
            )
=======
                    "all details",
                    "every",
                    "each ",
                    "how many",
                    "count",
                    "total number",
                    "entire",
                    "full list",
                ]
            ) or q.startswith("list")
>>>>>>> dc02b3a (updated gui)

            if broad_question:

                result_df = df

            elif matched_indices:

                matched_indices.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                indices = [
                    item[0]
                    for item in matched_indices
                ]

                result_df = df.loc[
                    indices
                ]

            else:

                # For small tables, send all rows.
                # This prevents missing information.
                if len(df) <= 100:

                    result_df = df

                else:

                    continue

            results.append(
                self.dataframe_to_text(
                    result_df,
                    source
                )
            )

        return results

    # =========================================================
    # BUILD FINAL PROMPT
    # =========================================================

    def build_table_prompt(
        self,
        query,
        table_context,
        chat_history=None
    ):

        history_text = ""

        if chat_history:

            history_text = (
                "\nPrevious conversation:\n"
            )

            for message in chat_history:

                role = message.get(
                    "role",
                    ""
                )

                content = message.get(
                    "content",
                    ""
                )

                history_text += (
                    f"{role}: "
                    f"{content}\n"
                )

        prompt = f"""
You are a RAG assistant.

Answer the user's question using ONLY the table data
provided below.

IMPORTANT RULES:

1. Do NOT invent employee names.
2. Do NOT invent salary values.
3. Do NOT invent performance ratings.
4. Do NOT omit matching records.
5. If the question asks for a list, return ALL matching records.
6. If the question asks for people below/above a value,
   carefully check every provided row.
7. Use the exact values from the table.
8. If a value is missing, say "Not provided".
9. Do not use SQL.
10. Do not assume information that is not present.

{history_text}

TABLE DATA
==========

{chr(10).join(table_context)}

USER QUESTION
=============

{query}

ANSWER
======

Provide a clear and complete answer.
"""

        return prompt

    # =========================================================
    # ASK
    # =========================================================

    def ask(
        self,
        query,
        chat_history=None
    ):

        query = str(
            query
        ).strip()

        if not query:

            return (
                "Please enter a question."
            )

        # =====================================================
        # TABLE QUESTION
        # =====================================================

        # Load tables once and reuse for both the detection
        # check and the actual retrieval, instead of reading
        # every CSV/Excel file from disk twice per question.
        tables = self.load_tables()

        if self.is_table_question(
            query,
            tables=tables
        ):

            print("\n")
            print("=" * 70)
            print("TABLE RAG QUESTION")
            print("=" * 70)

            print(
                f"Question : {query}"
            )

            table_context = (
                self.retrieve_from_tables(
                    query,
                    tables=tables
                )
            )

            if table_context:

                print(
                    "\nTable information found."
                )

                for i, context in enumerate(
                    table_context,
                    start=1
                ):

                    print(
                        f"\nTable Context {i}:"
                    )

                    print(
                        context
                    )

                prompt = (
                    self.build_table_prompt(
                        query,
                        table_context,
                        chat_history
                    )
                )

                print(
                    "\nGenerating answer..."
                )

                answer = (
                    self.llm.generate(
                        prompt
                    )
                )

                print(
                    "\nAnswer:"
                )

                print(
                    answer
                )

                return answer

            print(
                "No matching table data found."
            )

        # =====================================================
        # NORMAL PDF RAG
        # =====================================================

        if self.retriever is None:

            self.load_vector_database()

        print("\n")
        print("=" * 70)
        print("PDF RAG QUESTION")
        print("=" * 70)

        print(
            f"Question : {query}"
        )

        # =====================================================
        # RETRIEVE
        # =====================================================

        print(
            "\nRetrieving relevant information..."
        )

<<<<<<< HEAD
        retrieved_chunks = (
            self.retriever.retrieve(
                query
=======
        # Broad / "tell me everything" style questions need more
        # context than a narrow factual lookup. Widen top_k when
        # the query signals it wants comprehensive coverage.
        q_normalized = self.normalize_text(query)

        broad_signal_words = [
            "all", "every", "each", "entire", "complete",
            "full", "list", "summary", "summarize", "how many",
            "count", "total"
        ]

        is_broad = any(
            word in q_normalized
            for word in broad_signal_words
        )

        top_k = 15 if is_broad else 8

        retrieved_chunks = (
            self.retriever.retrieve(
                query,
                top_k=top_k
>>>>>>> dc02b3a (updated gui)
            )
        )

        print(
            "\nRetrieved Chunks:"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                f"\nChunk {i}:"
            )

            print(
                chunk
            )

        # =====================================================
        # BUILD PROMPT
        # =====================================================

        prompt = (
            self.prompt_builder.build_prompt(
                retrieved_chunks,
                query,
                chat_history
            )
        )

        # =====================================================
        # GEMINI
        # =====================================================

        print(
            "\nGenerating answer..."
        )

        answer = (
            self.llm.generate(
                prompt
            )
        )

        print(
            "\nAnswer:"
        )

        print(
            answer
        )

        return answer