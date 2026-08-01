from app.utils import load_pdf
from app.chunker import TextChunker
from app.embeddings import EmbeddingModel


class RAGPipeline:

    def __init__(self):

        self.chunker = TextChunker()

        self.embedding_model = EmbeddingModel()

    def build_index(self, pdf_path):

        print("Loading PDF...")

        text = load_pdf(pdf_path)

        print("Chunking document...")

        chunks = self.chunker.split_text(text)

        print(f"Total Chunks : {len(chunks)}")

        print("Generating Embeddings...")

        embeddings = self.embedding_model.create_embeddings(chunks)

        print("Saving Vector Database...")

        # FAISS code will be added later

        return chunks, embeddings
