# llm_answer.py
import streamlit as st
from openai import AzureOpenAI

def get_llm_client():
    """Returns configured Azure OpenAI client from session_state."""
    if (
        "AZURE_OPENAI_API_KEY" not in st.session_state 
        or "AZURE_OPENAI_ENDPOINT" not in st.session_state
        or not st.session_state.AZURE_OPENAI_API_KEY
        or not st.session_state.AZURE_OPENAI_ENDPOINT
    ):
        return None
    
    return AzureOpenAI( 
        api_key=st.session_state.AZURE_OPENAI_API_KEY,
        azure_endpoint=st.session_state.AZURE_OPENAI_ENDPOINT,
        api_version=st.session_state.AZURE_OPENAI_API_VERSION
    )


def generate_answer(question, chunks, max_tokens=300):
    client = get_llm_client()
    if client is None:
        return "⚠️ Azure OpenAI client not configured. Please check your API Key & Endpoint."

    if not chunks:
        return "No relevant information found in PDF."

    context = "\n\n".join([f"{c['chunk']} (Page {c['page']})" for c in chunks])

    prompt = f"""
Use ONLY the below PDF content to answer the question.

Question:
{question}

PDF Context:
{context}

Answer the user's question briefly and clearly.
"""

    response = client.chat.completions.create(
        model=st.session_state.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()
