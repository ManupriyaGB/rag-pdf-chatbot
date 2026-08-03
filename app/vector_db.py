from langchain_community.vectorstores import FAISS


class VectorStore:

    def __init__(self):
        self.db = None

    def create_vector_store(self, chunks, embedding_model):

        print("=" * 60)
        print("CREATING FAISS VECTOR DATABASE")
        print("=" * 60)

        print(f"Total Chunks : {len(chunks)}")

        self.db = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model.embeddings
        )

        print("\nFAISS Index Created Successfully")

        return self.db

    def save_vector_store(self, path="vector_db"):

        print("\nSaving FAISS Index...")

        self.db.save_local(path)

        print(f"Vector Database Saved in : {path}")

    def load_vector_store(
            self,
            embedding_model,
            path="vector_db"):

        print("\nLoading FAISS Database...")

        self.db = FAISS.load_local(
            path,
            embedding_model.embeddings,
            allow_dangerous_deserialization=True
        )

        print("FAISS Loaded Successfully")

        return self.db