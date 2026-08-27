"""Splits document text into overlapping chunks suitable for embedding."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split text into chunks by paragraph/sentence boundaries, respecting chunk_size."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    # Split on paragraph boundaries first, then greedily pack into chunks.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
            continue

        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself is too long: split by sentence.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sentence

    if current:
        chunks.append(current)

    # Apply overlap by prepending the tail of the previous chunk.
    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0 or chunk_overlap <= 0:
            overlapped.append(chunk)
        else:
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(f"{tail} {chunk}".strip())

    return [
        Chunk(text=c, source=source, chunk_index=i) for i, c in enumerate(overlapped)
    ]
