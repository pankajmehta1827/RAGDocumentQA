"""Upload documents page — add, process, and remove indexed files."""
from __future__ import annotations

import gc

import streamlit as st

from rag.chunker import chunk_text
from rag.document_loader import extract_text
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore

st.title("Upload documents")
st.caption("Upload PDF or DOCX files to index them for retrieval. Ask questions about them on the Retrieval information page.")

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
                "image-based PDF (or a design template that renders text as graphics). "
                "OCR was attempted automatically as a fallback; if it still failed, try "
                "re-exporting it as a text-based PDF/DOCX."
            )

st.divider()

store = st.session_state.vector_store
if store and store.sources:
    st.subheader("Indexed files")
    for name in store.sources:
        col1, col2 = st.columns([5, 1])
        col1.write(name)
        if col2.button(":material/delete:", key=f"remove_{name}", help=f"Remove {name}"):
            store.remove_source(name)
            if name in st.session_state.processed_files:
                st.session_state.processed_files.remove(name)
            st.rerun()
else:
    st.info("No documents indexed yet.")

if st.button("Clear all documents", type="secondary"):
    st.session_state.vector_store = None
    st.session_state.processed_files = []
    st.session_state.chat_history = []
    st.session_state.qa_log = []
    st.rerun()
