from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and return all extracted text.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text
