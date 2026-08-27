"""In-memory FAISS vector store for chunk retrieval."""
from __future__ import annotations

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized vectors
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, dimension), dtype="float32")

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) == 0:
            return
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors])

    def remove_source(self, source: str) -> None:
        """Drop all chunks from the given source file and rebuild the index."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.empty((0, self.dimension), dtype="float32")

        self.index = faiss.IndexFlatIP(self.dimension)
        if len(self.vectors) > 0:
            self.index.add(self.vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    @property
    def sources(self) -> list[str]:
        seen = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
