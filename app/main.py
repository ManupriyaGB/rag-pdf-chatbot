from utils import load_pdf
from chunker import TextChunker
from embeddings import EmbeddingModel

pdf_path = "data/raw/attention-is-all-you-need.pdf"

# Load PDF
text = load_pdf(pdf_path)

# Chunk text
chunker = TextChunker()
chunks = chunker.split_text(text)

print(f"Total Chunks : {len(chunks)}")

# Generate Embeddings
embedding_model = EmbeddingModel()

vectors = embedding_model.create_embeddings(chunks)

print(f"\nTotal Embeddings : {len(vectors)}")

print(f"\nEmbedding Dimension : {len(vectors[0])}")