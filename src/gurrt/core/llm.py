from gurrt.core.prompts import (LLM_QUERY_PROMPT, LLM_SYSTEM_PROMPT,
                                LOW_FIDELITY_VISUAL_NOTE)
from gurrt.core.context import format_prior_chat
from gurrt.core.debuglog import log_query
from gurrt.core import openrouter

from supermemory import Supermemory


def _fmt_ts(sec) -> str:
    if sec is None:
        return "??:??"
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def format_timeline(caption_list: list, asr_list: list) -> str:
    """Interleave what was shown and what was said, in time order.

    Two parallel blocks give the model no way to tell which caption belongs
    with which stretch of speech, so deictic transcript ("this term here")
    has no referent. One clock restores that link. Everything retrieved is
    included - trimming happens by retrieving less, not by cutting here.
    """
    events = []
    for f in caption_list:
        caption = (f.get("caption") or "").strip()
        if caption:
            events.append((f.get("start_sec") or 0.0, 0,
                           f"[{_fmt_ts(f.get('start_sec'))}] SHOWN: {caption}"))
    for a in asr_list:
        text = (a.get("text") or "").strip()
        if text:
            events.append((a.get("start_sec") or 0.0, 1,
                           f"[{_fmt_ts(a.get('start_sec'))}] SAID:  {text}"))
    events.sort(key=lambda e: (e[0], e[1]))
    return "\n".join(line for _, _, line in events)


class LLMService:
    def __init__(self, settings):
        self.settings = settings
        self.client_memory = Supermemory(api_key=settings.SUPERMEMORY_API_KEY)

    async def query_llm(self,
                        query: str,
                        timeline: str,
                        low_fidelity_visual: bool = False) -> str:
        chat_context = self.client_memory.search.documents(
            q=query,
            container_tags=["Previous_Chat"],
            limit=3,
        )
        previous_chat = format_prior_chat(chat_context)

        system_prompt = LLM_SYSTEM_PROMPT
        if low_fidelity_visual:
            system_prompt += LOW_FIDELITY_VISUAL_NOTE

        # System carries the rules, the user message carries the evidence, and
        # the question comes last: with a long context block, a query buried
        # at the top gets attended to far less than one at the end.
        user_prompt = LLM_QUERY_PROMPT.format(
            timeline=timeline or "No indexed content matched this question.",
            previous_chat=previous_chat,
            query=query,
        )

        result = await openrouter.chat(self.settings, system_prompt, user_prompt)

        log_query(
            self.settings,
            query=query,
            system_prompt=system_prompt,
            timeline=timeline,
            previous_chat=previous_chat,
            rendered_human=user_prompt,
            answer=result,
            low_fidelity_visual=low_fidelity_visual,
        )

        context = f"{query}\n\n\n{result}"
        self.client_memory.add(
            content=context,
            container_tags=["Previous_Chat"],
            metadata={"note_id": "Retrieved Chat"},
        )
        return result

    def delete(self) -> dict:
        chat_deleted = self.client_memory.documents.delete_bulk(
            container_tags=["Previous_Chat"])
        return {"chat_deleted": chat_deleted}
