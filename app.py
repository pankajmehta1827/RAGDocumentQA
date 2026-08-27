"""RAG Document Q&A — entry point and page router."""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon=":material/description:", layout="wide")

# --- Session state -----------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []  # list of dicts with question/answer/sources for display

pages = [
    st.Page("app_pages/upload.py", title="Upload documents", icon=":material/upload_file:"),
    st.Page("app_pages/retrieval.py", title="Retrieval information", icon=":material/manage_search:"),
]
st.navigation(pages).run()
