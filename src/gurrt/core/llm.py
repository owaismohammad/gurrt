from gurrt.core.prompts import LLM_QUERY_PROMPT, LLM_SYSTEM_PROMPT
from gurrt.core.context import format_prior_chat

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supermemory import Supermemory


class LLMService:
    def __init__(self, settings):
        self.settings = settings
        self.llm = ChatGroq(model = settings.LLM_MODEL,
                            api_key= settings.GROQ_API_KEY,
                            max_tokens= 4096,
                        )
        self.client_memory = Supermemory(api_key=settings.SUPERMEMORY_API_KEY)

    async def query_llm(self,
                        query:str,
                        timeline: str) -> str:
        chat_context = self.client_memory.search.documents(
            q= query,
            container_tags = ["Previous_Chat"],
            limit = 3
        )
        previous_chat = format_prior_chat(
            chat_context, self.settings.CHAT_TOKEN_BUDGET)

        parser = StrOutputParser()
        # System carries the rules, human carries the evidence, and the question
        # comes last: with a long context block, a query buried at the top gets
        # attended to far less than one at the end.
        prompt = ChatPromptTemplate.from_messages([
            ("system", LLM_SYSTEM_PROMPT),
            ("human", LLM_QUERY_PROMPT),
        ])
        chain = prompt | self.llm | parser
        result = await chain.ainvoke({
            "timeline": timeline or "No indexed content matched this question.",
            "previous_chat": previous_chat,
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

