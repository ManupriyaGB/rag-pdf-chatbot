class Retriever:
    """
    Responsible for retrieving relevant chunks
    from the vector database.
    """

    def __init__(self):
        self.vector_db = None

    def load_vector_database(self):
        """
        Load FAISS index from disk.
        """
        pass

    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k most similar chunks.
        """

        print(f"Searching for: {query}")

        # FAISS similarity search will be added later

        return []
