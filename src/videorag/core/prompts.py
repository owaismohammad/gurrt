BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_QUERY_PROMPT = """
You are an AI assistant answering questions based on a video.

You are given:
1. Context extracted from video frames (visual information).
2. Context extracted from audio transcripts (spoken information).
3. A user query.

Your task:
- Answer the query using ONLY the provided context.
- Combine visual and audio context when helpful.
- If the context is insufficient, clearly say:
  "The provided video context does not contain enough information to answer this question."
- Do NOT make up facts.
- Keep the answer clear, concise, and relevant.

---------------------
VIDEO FRAME CONTEXT:
{context_frame}

---------------------
AUDIO TRANSCRIPT CONTEXT:
{context_audio}

---------------------
USER QUERY:
{query}

---------------------
ANSWER:
"""
