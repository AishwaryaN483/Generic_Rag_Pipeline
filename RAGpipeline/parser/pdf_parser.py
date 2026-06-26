from pypdf import PdfReader
from langchain_core.documents import Document


def parse_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path,
                "type": "pdf"
            }
        )
    ]