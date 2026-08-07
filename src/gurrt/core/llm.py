from gurrt.core.prompts import LLM_QUERY_PROMPT

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supermemory import Supermemory

def _fmt_ts(sec) -> str:
    if sec is None:
        return "??:??"
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def format_timeline(caption_list: list, asr_list: list) -> str:
    """Interleave what was shown and what was said into one time-ordered block.

    Two parallel blocks give the model no way to tell which caption goes with
    which stretch of speech, so deictic transcript ("this term here") has no
    referent. Sorting both streams onto one clock restores that link.
    """
    events = []
    for f in caption_list:
        events.append((f.get("start_sec") or 0.0,
                       f"[{_fmt_ts(f.get('start_sec'))}] SHOWN: {f['caption']}"))
    for a in asr_list:
        events.append((a.get("start_sec") or 0.0,
                       f"[{_fmt_ts(a.get('start_sec'))}] SAID:  {a['text']}"))
    events.sort(key=lambda e: e[0])
    return "\n".join(text for _, text in events)


class LLMService:
    def __init__(self, settings):
        self.llm = ChatGroq(model = settings.LLM_MODEL,
                            api_key= settings.GROQ_API_KEY,
                            max_tokens= 4096,
                        )
        self.client_memory = Supermemory(api_key=settings.SUPERMEMORY_API_KEY)

    async def query_llm(self, 
                        query:str,
                        caption_list: list,
                        asr_list: list) -> str:
        timeline = format_timeline(caption_list, asr_list)
        chat_context = self.client_memory.search.documents(
            q= query,
            container_tags = ["Previous_Chat"],
            limit = 1
        )
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template = LLM_QUERY_PROMPT,
            input_variables = ["timeline", "previous_chat","query"]
        )
        chain = prompt | self.llm | parser
        result = await chain.ainvoke({
            "timeline": timeline,
            "previous_chat": chat_context,
            "query" : query
        })
        context = f"{query}\n\n\n{result}"
        self.client_memory.add(
            content = context,
            container_tags = ["Previous_Chat"],
            metadata = {
                "note_id": "Retrieved Chat"
            }
        )
        return result

    def delete(self) -> dict:
        chat_deleted = self.client_memory.documents.delete_bulk(container_tags=["Previous_Chat"])
        return {"chat_deleted": chat_deleted}

