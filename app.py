"""RAG Document Q&A — upload a PDF/DOCX, ask questions, get answers from Groq."""
from __future__ import annotations

import gc
import os

import streamlit as st

from rag.chunker import chunk_text
from rag.document_loader import extract_text
from rag.embeddings import EmbeddingModel
from rag.llm import generate_answer, list_models
from rag.vector_store import VectorStore

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# Static fallback shown before a key is entered, or if the live lookup fails.
# The dropdown is replaced with the key's real available models once possible,
# since Groq periodically retires/restricts model ids.
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_models(key: str) -> list[str]:
    try:
        return list_models(key)
    except Exception:
        return []

# --- Session state ---------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []  # list of dicts with question/answer/sources for display

# --- Sidebar: configuration --------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com. Stored only for this session.",
    )
    live_models = _fetch_models(api_key) if api_key else []
    model_options = live_models or FALLBACK_MODELS
    if api_key and not live_models:
        st.caption("Couldn't fetch your available models; showing common defaults.")
    model = st.selectbox("Groq model", model_options, index=0)
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)

    st.divider()
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", disabled=not uploaded_files):
        embedder = EmbeddingModel()
        store = st.session_state.vector_store or VectorStore(embedder.dimension)

        new_files = [
            f for f in uploaded_files if f.name not in st.session_state.processed_files
        ]
        if not new_files:
            st.info("These files were already processed.")
        else:
            progress = st.progress(0.0, text="Processing documents...")
            empty_files = []
            for i, file in enumerate(new_files):
                text = extract_text(file.getvalue(), file.name)
                chunks = chunk_text(text, source=file.name)
                if chunks:
                    vectors = embedder.embed([c.text for c in chunks])
                    store.add(vectors, chunks)
                    st.session_state.processed_files.append(file.name)
                    del vectors
                else:
                    empty_files.append(file.name)
                progress.progress((i + 1) / len(new_files), text=f"Processed {file.name}")

                # OCR (torch + easyocr) on large scanned PDFs is memory-heavy;
                # freeing each file's intermediates before the next iteration
                # keeps peak memory from stacking up across a multi-file batch.
                del text, chunks
                gc.collect()
            progress.empty()
            st.session_state.vector_store = store

            indexed_count = len(new_files) - len(empty_files)
            if indexed_count:
                st.success(f"Indexed {indexed_count} file(s), {len(store.chunks)} chunks total.")
            for name in empty_files:
                st.warning(
                    f"No extractable text found in **{name}**. It's likely a scanned or "
                    "image-based PDF (or a design template that renders text as graphics), "
                    "and OCR tools (Tesseract + Poppler) aren't installed. Try re-exporting "
                    "it as a text-based PDF/DOCX, or install Tesseract OCR and Poppler to "
                    "enable automatic OCR fallback."
                )

    store = st.session_state.vector_store
    if store and store.sources:
        st.caption("Indexed files:")
        for name in store.sources:
            col1, col2 = st.columns([4, 1])
            col1.caption(name)
            if col2.button("🗑️", key=f"remove_{name}", help=f"Remove {name}"):
                store.remove_source(name)
                if name in st.session_state.processed_files:
                    st.session_state.processed_files.remove(name)
                st.rerun()

    if st.button("Clear session", type="secondary"):
        st.session_state.vector_store = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.session_state.qa_log = []
        st.rerun()

# --- Main: chat interface ----------------------------------------------------
st.title("📄 RAG Document Q&A")
st.caption("Upload a PDF or DOCX in the sidebar, then ask questions about its content. Answers are generated by Groq using only retrieved excerpts from your document.")

for turn in st.session_state.qa_log:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["sources"]:
            with st.expander("Sources"):
                for text, source, score in turn["sources"]:
                    st.markdown(f"**{source}** (similarity: {score:.2f})")
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))

question = st.chat_input("Ask a question about your document...")

if question:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
    elif not st.session_state.vector_store or st.session_state.vector_store.is_empty:
        st.error("Upload and process at least one document first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant excerpts and generating answer..."):
                embedder = EmbeddingModel()
                query_vector = embedder.embed([question])[0]
                results = st.session_state.vector_store.search(query_vector, top_k=top_k)
                context_chunks = [(c.text, c.source, score) for c, score in results]

                try:
                    answer = generate_answer(
                        api_key=api_key,
                        model=model,
                        question=question,
                        context_chunks=context_chunks,
                        chat_history=st.session_state.chat_history[-6:],
                    )
                except Exception as e:
                    answer = f"Error calling Groq API: {e}"

            st.markdown(answer)
            if context_chunks:
                with st.expander("Sources"):
                    for text, source, score in context_chunks:
                        st.markdown(f"**{source}** (similarity: {score:.2f})")
                        st.text(text[:500] + ("..." if len(text) > 500 else ""))

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.qa_log.append(
            {"question": question, "answer": answer, "sources": context_chunks}
        )
