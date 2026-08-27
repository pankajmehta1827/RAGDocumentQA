"""Groq LLM client for answer generation over retrieved context."""
from __future__ import annotations

from groq import Groq

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "document excerpts. If the answer is not contained in the excerpts, say you "
    "don't know based on the document. Cite the source filename when relevant. "
    "Be concise and accurate."
)


def build_prompt(question: str, context_chunks: list[tuple[str, str, float]]) -> str:
    """context_chunks: list of (text, source, score)."""
    context_blocks = []
    for i, (text, source, _score) in enumerate(context_chunks, start=1):
        context_blocks.append(f"[Excerpt {i} — {source}]\n{text}")
    context_str = "\n\n".join(context_blocks) if context_blocks else "(no relevant excerpts found)"

    return (
        f"Document excerpts:\n\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the information above."
    )


def list_models(api_key: str) -> list[str]:
    """Return chat-capable model ids actually available to this API key."""
    client = Groq(api_key=api_key)
    response = client.models.list()
    excluded = ("whisper", "tts")  # audio models, not usable for chat completions
    ids = [
        m.id
        for m in response.data
        if getattr(m, "active", True) and not any(x in m.id for x in excluded)
    ]
    return sorted(ids)


def generate_answer(
    api_key: str,
    model: str,
    question: str,
    context_chunks: list[tuple[str, str, float]],
    chat_history: list[dict] | None = None,
) -> str:
    client = Groq(api_key=api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": build_prompt(question, context_chunks)})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content
