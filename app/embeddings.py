class EmbeddingModel:
    """
    Responsible for converting text into embeddings.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """
        Loads the embedding model.
        (Implementation will be added after installing packages)
        """
        pass

    def create_embeddings(self, chunks):
        """
        Converts text chunks into embeddings.
        """
        pass

    def create_query_embedding(self, query):
        """
        Converts a user query into an embedding.
        """
        pass
