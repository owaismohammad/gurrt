"""Prior-conversation handling for the ask prompt."""


def format_prior_chat(chat_context) -> str:
    """Render prior turns as plain text.

    The memory client returns a response object; interpolating it straight
    into a prompt dumps an SDK repr of ids and scores, which is pure noise.
    The number of turns is already bounded by the search call's limit.
    """
    if not chat_context:
        return "None."

    docs = getattr(chat_context, "results", None) or getattr(chat_context, "documents", None)
    if docs is None and isinstance(chat_context, dict):
        docs = chat_context.get("results") or chat_context.get("documents")
    if docs is None:
        docs = chat_context if isinstance(chat_context, list) else []

    texts = []
    for d in docs:
        content = (getattr(d, "content", None)
                   or getattr(d, "memory", None)
                   or (d.get("content") if isinstance(d, dict) else None)
                   or (d.get("memory") if isinstance(d, dict) else None))
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())

    return "\n\n".join(texts) if texts else "None."
