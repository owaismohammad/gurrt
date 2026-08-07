from gurrt.core.prompts import (LLM_QUERY_PROMPT, LLM_SYSTEM_PROMPT,
                                LOW_FIDELITY_VISUAL_NOTE)
from gurrt.core.context import format_prior_chat, estimate_tokens
from gurrt.core.debuglog import log_query
from gurrt.cli import ui

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supermemory import Supermemory


class LLMService:
    def __init__(self, settings):
        self.settings = settings
        self.llm = ChatGroq(model = settings.LLM_MODEL,
                            api_key= settings.GROQ_API_KEY,
                            max_tokens= settings.MAX_OUTPUT_TOKENS,
                        )
        self.client_memory = Supermemory(api_key=settings.SUPERMEMORY_API_KEY)

    def _output_allowance(self, system_prompt: str, rendered_human: str) -> int:
        """How many output tokens we can reserve without blowing the TPM cap.

        Groq counts input + max_tokens against the same per-minute allowance,
        so a large max_tokens is spent whether the answer uses it or not. Size
        the reservation to what is actually left.
        """
        estimated_input = (estimate_tokens(system_prompt)
                           + estimate_tokens(rendered_human))
        headroom = (self.settings.TPM_LIMIT
                    - estimated_input
                    - self.settings.TPM_SAFETY_MARGIN)
        allowance = min(self.settings.MAX_OUTPUT_TOKENS, headroom)

        if allowance < self.settings.MIN_OUTPUT_TOKENS:
            # Context alone is crowding out the answer. Ask anyway with a
            # usable floor, but say so - the fix is a smaller context budget.
            ui.warn(f"Context (~{estimated_input} tokens) leaves only "
                    f"{max(0, headroom)} for the answer; requesting "
                    f"{self.settings.MIN_OUTPUT_TOKENS}. Lower "
                    f"CONTEXT_TOKEN_BUDGET if this keeps happening.")
            return self.settings.MIN_OUTPUT_TOKENS
        return allowance

    async def query_llm(self,
                        query:str,
                        timeline: str,
                        low_fidelity_visual: bool = False) -> str:
        chat_context = self.client_memory.search.documents(
            q= query,
            container_tags = ["Previous_Chat"],
            limit = 3
        )
        previous_chat = format_prior_chat(
            chat_context, self.settings.CHAT_TOKEN_BUDGET)

        system_prompt = LLM_SYSTEM_PROMPT
        if low_fidelity_visual:
            system_prompt += LOW_FIDELITY_VISUAL_NOTE

        parser = StrOutputParser()
        # System carries the rules, human carries the evidence, and the question
        # comes last: with a long context block, a query buried at the top gets
        # attended to far less than one at the end.
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", LLM_QUERY_PROMPT),
        ])
        variables = {
            "timeline": timeline or "No indexed content matched this question.",
            "previous_chat": previous_chat,
            "query" : query
        }
        rendered_human = LLM_QUERY_PROMPT.format(**variables)

        max_out = self._output_allowance(system_prompt, rendered_human)
        chain = prompt | self.llm.bind(max_tokens=max_out) | parser
        result = await chain.ainvoke(variables)

        log_query(
            self.settings,
            query=query,
            system_prompt=system_prompt,
            timeline=variables["timeline"],
            previous_chat=previous_chat,
            rendered_human=rendered_human,
            answer=result,
            max_output_tokens=max_out,
            low_fidelity_visual=low_fidelity_visual,
        )

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

