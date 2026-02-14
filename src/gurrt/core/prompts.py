BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_QUERY_PROMPT = """
Answer the question using only the information available below.

Rules:
- Respond with a direct answer only.
- Do NOT mention the context.
- Do NOT explain reasoning.
- Do NOT state whether information is sufficient.
- If the answer cannot be determined, respond with:
  Not enough information.

---------------------
VIDEO:
{context_frame}

AUDIO:
{context_audio}

QUESTION:
{query}

ANSWER:
"""

