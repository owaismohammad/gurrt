BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_QUERY_PROMPT = """You are a knowledgeable assistant answering questions about a video lecture.

Below is an excerpt from the lecture timeline, in chronological order. SHOWN lines
are what was on screen (slides, board, diagrams); SAID lines are what the lecturer
said at that moment. Lines close together in time refer to the same thing, so use
a SHOWN line to resolve vague speech like "this term here".

Answer directly. Cite the timestamp you drew from, like (04:12). If the timeline
below does not cover the question, say so rather than guessing.

LECTURE TIMELINE:
{timeline}

PRIOR CONVERSATION:
{previous_chat}

QUESTION: {query}
ANSWER:"""

VLM_PROMPT = """Describe all visible text,
                equations, diagrams and symbols.
                Ignore appearance and background."""

GEMMA_CAPTION_PROMPT = """You are indexing a frame from an educational video for a search engine.

Describe ONLY what is actually visible. Never guess at content you cannot see.

Reply with exactly these six lines, in this order, and nothing else:

SCENE: one of [slide, slide+speaker, whiteboard, whiteboard+speaker, closeup-writing, code, terminal, speaker-only, other]
TITLE: largest heading, copied exactly. NONE if absent.
TEXT: every readable word of body text, bullets, labels, axis names, in reading order, separated by " | ". NONE if absent.
MATH: every equation or symbolic expression, copied exactly. NONE if absent.
FIGURE: diagram/graph/table type + its labelled parts + what it shows, under 30 words. NONE if absent.
TOPIC: the one concept this frame teaches, under 15 words.

Rules:
- TEXT and MATH are transcriptions, not summaries. Copy the words exactly. Do not paraphrase, shorten, or add words of your own.
- If text is visible but too blurry to read, write UNREADABLE for that field.
- Never describe the person, their clothing, the room, lighting, or background."""
