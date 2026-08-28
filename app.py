"""RAG Document Q&A — entry point and page router."""
from __future__ import annotations

import os

# PyTorch (via sentence-transformers and easyocr) defaults to using every
# CPU core for its thread pool. On a machine that's also running a browser
# and OS UI, that causes system-wide sluggishness during any embedding or
# OCR call. These must be set before torch is imported anywhere in the
# process, so they're set here, at the very top of the entry point.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

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
