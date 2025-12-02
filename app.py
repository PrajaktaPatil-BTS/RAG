# app.py
import streamlit as st
from chunker import load_pdf, chunk_text
from retriever import PDFRetriever
from llm_answer import generate_answer


# ---------------------------------------------------------
# Chat Message Styling
# ---------------------------------------------------------
def message_box(text, sender="assistant"):
    text = text.replace("\n", "<br>")
    
    if sender == "assistant":
        st.markdown(
            f"""
            <div style="background:#333;color:white;padding:12px;border-radius:10px;
                        max-width:80%;margin:8px 0;">
                🤖 <b>Bot:</b><br>{text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background:#4CAF50;color:white;padding:12px;border-radius:10px;
                        max-width:80%;margin-left:auto;margin:8px 0;">
                🙋 <b>You:</b><br>{text}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# MAIN Streamlit APP
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")

    st.title("📘 PDF Chatbot")

    # ---------------------------------------------------------
    # SIDEBAR  (Always visible)
    # ---------------------------------------------------------
    st.sidebar.header("🔐 Azure OpenAI Credentials")

    # Ensure persistent storage
    if "AZURE_OPENAI_ENDPOINT" not in st.session_state:
        st.session_state.update({
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_API_KEY": "",
            "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4.1-mini"
        })

    st.session_state.AZURE_OPENAI_ENDPOINT = st.sidebar.text_input(
        "Endpoint", value=st.session_state.AZURE_OPENAI_ENDPOINT
    )

    st.session_state.AZURE_OPENAI_API_KEY = st.sidebar.text_input(
        "API Key", type="password", value=st.session_state.AZURE_OPENAI_API_KEY
    )

    st.session_state.AZURE_OPENAI_API_VERSION = st.sidebar.text_input(
        "API Version", value=st.session_state.AZURE_OPENAI_API_VERSION
    )

    st.session_state.AZURE_OPENAI_DEPLOYMENT = st.sidebar.text_input(
        "Deployment Name", value=st.session_state.AZURE_OPENAI_DEPLOYMENT
    )

    # ---------------------------------------------------------
    # PDF Upload (Sidebar)
    # ---------------------------------------------------------
    st.sidebar.header("📄 Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if uploaded_file and st.session_state.retriever is None:
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pages = load_pdf(pdf_path)
        chunks = chunk_text(pages)

        retriever = PDFRetriever()
        retriever.build_index(chunks)
        st.session_state.retriever = retriever

        st.sidebar.success("PDF processed successfully!")

    # ---------------------------------------------------------
    # CHAT UI
    # ---------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show past messages
    for msg in st.session_state.messages:
        message_box(msg["content"], msg["role"])

    # Input box
    user_input = st.chat_input("Ask something from the PDF…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        message_box(user_input, "user")

        # Search PDF chunks
        top_chunks = st.session_state.retriever.search(user_input)

        # LLM Answer
        answer = generate_answer(user_input, top_chunks)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        message_box(answer, "assistant")


# Run App
if __name__ == "__main__":
    main()
