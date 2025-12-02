"""
main.py

- Orchestrates PDF parsing, chunking, FAISS index, and GPT answer loop.
- On first run, builds index. Subsequent runs load index from disk.
"""

import os
from chunker import load_pdf, chunk_text
from retriever import PDFRetriever
from llm_answer import generate_answer

PDF_FILE = "Vridhi Home Finance.pdf"  
INDEX_FILES_EXIST = all(os.path.exists(f) for f in ["faiss_tfidf.index", "tfidf_vectorizer.pkl", "chunks.txt"])

retriever = PDFRetriever()

if INDEX_FILES_EXIST:
    print("Loading existing index...")
    retriever.load_index()
else:
   # read data from pdf
    text = load_pdf(PDF_FILE)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    retriever.build_index(chunks)
    print(f"Index built with {len(chunks)} chunks.")

print("\nPDF Chatbot ready. Type questions (or 'exit'):")

while True:
    query = input("\nQuestion: ").strip()
    if query.lower() in ("exit", "quit"):
        break
    if not query:
        continue

    top_chunks = retriever.search(query, top_k=3)
    if not top_chunks:
        print("No relevant information found.")
        continue

    answer = generate_answer(query, top_chunks)
    print("\n--- Answer ---\n")
    print(answer)
