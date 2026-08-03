from langchain_community.vectorstores import FAISS


class Retriever:

    def __init__(self, vector_db):

        self.vector_db = vector_db

    def retrieve(self, query, k=3):

        print("=" * 70)
        print("RETRIEVER")
        print("=" * 70)

        print(f"\nUser Query :\n{query}")

        print("\nSearching Similar Chunks...")

        docs = self.vector_db.similarity_search(query, k=k)

        print(f"\nTop {k} Chunks Retrieved")

        print("=" * 70)

        for i, doc in enumerate(docs):

            print(f"\nChunk {i+1}")

            print("-" * 50)

            print(doc.page_content)

        return docs