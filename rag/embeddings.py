"""Local embedding model wrapper (no external API needed)."""
from __future__ import annotations

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


class EmbeddingModel:
    """Thin wrapper around a cached sentence-transformers model."""

    def __init__(self) -> None:
        self.model = _load_model()
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
