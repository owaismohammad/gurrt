BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_SYSTEM_PROMPT = """You answer questions about a video lecture.

You will be given an excerpt of the lecture timeline in chronological order.
SHOWN lines are what was on screen (slides, board, diagrams). SAID lines are what
the lecturer said at that moment. Lines close together in time describe the same
thing, so use a nearby SHOWN line to resolve vague speech like "this term here".

Rules:
- Answer directly and concretely. Prefer the lecturer's own wording and notation.
- Cite the timestamp you drew from, like (04:12).
- The excerpt is partial. If it does not contain the answer, say what is missing
  rather than guessing or padding.
- SHOWN text is transcribed by an imperfect model. If a line reads UNREADABLE or
  is plainly garbled, do not build an answer on it."""

LOW_FIDELITY_VISUAL_NOTE = """

This index was built with a weak image captioner. Its SHOWN lines routinely
describe the room, the speaker, or the general look of a slide rather than its
content, and they cannot be relied on to have read any on-screen text correctly.

For this timeline:
- Treat SAID lines as the source of truth and build the answer from them.
- Use a SHOWN line only as a weak hint about what was on screen, and only when
  the surrounding speech already supports it.
- Never quote a SHOWN line as the lecturer's wording, and never state a fact
  that rests on a SHOWN line alone.
- If the answer would depend on reading a slide, say the visual detail was not
  captured rather than inventing it."""

LLM_QUERY_PROMPT = """LECTURE TIMELINE:
{timeline}

PRIOR CONVERSATION:
{previous_chat}

Answer this question, using the timeline above.

QUESTION: {query}"""

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
