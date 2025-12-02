import PyPDF2

#Opens a PDF Reads every page Extracts text from each page Returns a list of pages with their text and page number
def load_pdf(pdf_path: str):
    """Load PDF text with page numbers."""
    pages = []

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i})

    return pages


#Takes the list of pages returned by load_pdf() Breaks long page text 
#into smaller chunks (500 words by default) Keeps the original page number for each chunk Returns a list of chunk dictionaries
def chunk_text(pages, chunk_size=500, overlap=50):
    """Split PDF text into chunks while keeping page numbers."""
    chunks = []

    for p in pages:
        words = p["text"].split()
        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append({"chunk": chunk, "page": p["page"]})

    return chunks
