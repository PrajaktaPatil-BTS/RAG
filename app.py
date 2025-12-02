import streamlit as st
from chunker import load_pdf, chunk_text
from retriever import PDFRetriever
from llm_answer import generate_answer
import os

# --- Page config ---
st.set_page_config(page_title="Vridhi Home Finance Chatbot", page_icon="🤖", layout="wide")

# --- Dark theme CSS + Chat bubbles ---
st.markdown("""
<style>
body, .stApp {
    background-color: #0C0C0C;
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}

.chat-container {
    height: 600px;
    overflow-y: auto;
    padding: 15px;
    border-radius: 10px;
    background-color: #1E1E1E;
    border: 1px solid #333;
}

.user-message {
    background-color: #4CAF50;
    color: #FFFFFF;
    padding: 12px 18px;
    border-radius: 20px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 10px;
    font-size: 16px;
}

.bot-message {
    background-color: #333333;
    color: #FFFFFF;
    padding: 12px 18px;
    border-radius: 20px;
    max-width: 70%;
    margin-right: auto;
    margin-bottom: 10px;
    font-size: 16px;
}

.bot-message ul {
    margin: 5px 0;
    padding-left: 20px;
}

.sources {
    font-size: 12px;
    color: #AAAAAA;
    margin-top: 5px;
}

input, .stTextInput>div>div>input {
    background-color: #1E1E1E;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns([1, 3])

# --- Left column: PDF Upload ---
with col1:
    st.markdown("<h3 style='color:white;'>Upload PDF</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file:
        pdf_path = uploaded_file.name
        with open(pdf_path, "wb") as f:
            #eturns the binary content of the uploaded PDF. and it locally
            f.write(uploaded_file.getbuffer())
        st.success("PDF uploaded successfully!")

        # object create retriever
        retriever = PDFRetriever()
        index_exists = os.path.exists("faiss_tfidf.index") and os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("chunks_with_pages.pkl")

        if index_exists:
            retriever.load_index()
        else:
            pages = load_pdf(pdf_path)
            chunks = chunk_text(pages, chunk_size=500, overlap=50)
            retriever.build_index(chunks)
            st.info(f"Index created with {len(chunks)} chunks.")

        st.session_state.retriever = retriever

# --- Right column: Chat ---
with col2:
    st.markdown("<h3 style='color:white;'>💬 Vridhi Home Finance Chatbot</h3>", unsafe_allow_html=True)

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat messages in chronological order
    st.markdown('<div class="chat-container" id="chat">', unsafe_allow_html=True)
    for chat in st.session_state.history:
        # User bubble
        st.markdown(
            f'<div class="user-message"><b>You:</b> {chat["question"]}</div>', unsafe_allow_html=True
        )
        # Bot bubble with formatted answer
        st.markdown(
            f'<div class="bot-message"><b>Bot:</b><br>{chat["answer"]}'
            f'<div class="sources">Sources: {", ".join(map(str, chat["sources"]))}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- User input with callback ---
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    def handle_input():
        user_question = st.session_state.user_input
        if user_question and "retriever" in st.session_state:
            top_chunks = st.session_state.retriever.search(user_question, top_k=3)
            raw_answer = generate_answer(user_question, top_chunks)

            # Format numbered lists automatically
            formatted_answer = raw_answer
            if any(f"{i}." in raw_answer for i in range(1, 10)):
                parts = raw_answer.split(' ')
                lines = raw_answer.split(' 1.')
                formatted_answer = "<ul>"
                for i, line in enumerate(lines):
                    if line.strip():
                        formatted_answer += f"<li>{line.strip()}</li>"
                formatted_answer += "</ul>"

            st.session_state.history.append({
                "question": user_question,
                "answer": formatted_answer,
                "sources": [c["page"] for c in top_chunks]
            })

            # Clear input box
            st.session_state.user_input = ""

    st.text_input("Ask a question:", key="user_input", on_change=handle_input)

    # Auto-scroll to bottom
    st.markdown("""
    <script>
    var chatDiv = window.parent.document.querySelector(".chat-container");
    if(chatDiv){ chatDiv.scrollTop = chatDiv.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)
