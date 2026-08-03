import faiss
import pickle
import numpy as np


class Retriever:

    def __init__(self, embedding_model):

        print("=" * 70)
        print("Initializing Retriever...")
        print("=" * 70)

        self.embedding_model = embedding_model

        print("\nLoading FAISS Index...")

        self.index = faiss.read_index(
            "vector_db/index.faiss"
        )

        print("FAISS Index Loaded Successfully")

        print("\nLoading Chunks...")

        with open("vector_db/chunks.pkl", "rb") as file:
            self.chunks = pickle.load(file)

        print(f"Total Chunks Loaded : {len(self.chunks)}")

    # --------------------------------------------------------
    # Retrieve Similar Chunks
    # --------------------------------------------------------

    def retrieve(self, query, top_k=3):

        print("\n" + "=" * 70)
        print("RETRIEVAL STARTED")
        print("=" * 70)

        print("\nUser Query :")
        print(query)

        # ---------------------------------------------
        # Query Embedding
        # ---------------------------------------------

        print("\nGenerating Query Embedding...")

        query_embedding = self.embedding_model.create_query_embedding(query)

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        print("Embedding Shape :", query_embedding.shape)

        # ---------------------------------------------
        # Similarity Search
        # ---------------------------------------------

        print("\nSearching FAISS...")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        print("\nSearch Completed")

        print("\nDistances")

        print(distances)

        print("\nIndices")

        print(indices)

        # ---------------------------------------------
        # Retrieve Chunks
        # ---------------------------------------------

        retrieved_chunks = []

        print("\n" + "=" * 70)
        print("TOP MATCHING CHUNKS")
        print("=" * 70)

        for rank, idx in enumerate(indices[0]):

            chunk = self.chunks[idx]

            retrieved_chunks.append(chunk)

            print(f"\nRank : {rank + 1}")

            print(f"Chunk Index : {idx}")

            print(f"Distance : {distances[0][rank]:.4f}")

            print("-" * 70)

            print(chunk)

            print("-" * 70)

        print("\nRetrieval Completed")

        return retrieved_chunks