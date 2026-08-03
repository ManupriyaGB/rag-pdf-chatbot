import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag import RAGPipeline


def main():

    print("=" * 70)
    print("RAG PDF CHATBOT")
    print("=" * 70)

    rag = RAGPipeline()

    pdf_path = "data/attention_is_all_you_need.pdf"

    # Build vector database if it doesn't exist
    if not os.path.exists("vector_db/index.faiss"):

        print("\nVector Database not found.")
        print("Creating Knowledge Base...\n")

        rag.build_vector_database(pdf_path)

    rag.load_database()

    while True:

        print("\n" + "=" * 70)

        query = input("Ask Question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = rag.ask(query)

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(answer)


if __name__ == "__main__":
    main()