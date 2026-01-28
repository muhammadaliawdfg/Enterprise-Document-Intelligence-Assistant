from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Returns:
    [
        {
            "page_number": 1,
            "text": "..."
        }
    ]
    """
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page_number": i + 1,
                "text": text
            })

    return pages
