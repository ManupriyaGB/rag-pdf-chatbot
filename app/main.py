from app.utils import load_pdf
from app.chunker import TextChunker

pdf_path = "data/raw/attention_is_all_you_need.pdf"

text = load_pdf(pdf_path)

chunker = TextChunker()

chunks = chunker.split_text(text)

print(f"Total Chunks : {len(chunks)}")

print()

print(chunks[0])
