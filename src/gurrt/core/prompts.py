BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_QUERY_PROMPT = """You are a knowledgeable assistant answering questions about a video lecture. Answer directly and confidently. Do not say "based on the transcript" or "it appears" — just answer.

AUDIO TRANSCRIPT (primary — ground truth):
{context_audio}

VISUAL CAPTIONS (secondary — use only if relevant to the audio; ignore if unrelated):
{context_frame}

PRIOR CONVERSATION:
{previous_chat}

QUESTION: {query}
ANSWER:"""

VLM_PROMPT = """Describe all visible text,
                equations, diagrams and symbols.
                Ignore appearance and background."""

GEMMA_CAPTION_PROMPT =  """Analyze this video lecture frame for a search indexing engine. 
                            Provide**On-Screen Content**: [Transcribe  any key text, equations, bullet points, or diagrams visible].
                            Be concise to the point No prose.No formatting.No introductions.
                            No explanations"""
