from videorag.core.prompts import LLM_QUERY_PROMPT

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supermemory import Supermemory

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
        
        context_caption = "\n".join(caption_list)
        asr_text = "\n".join(asr_list)
        
        self.client_memory.add(
            content = context_caption,
            container_tags = ["Frame_Captions"],
            metadata = {
                "note_id": "Retrieved Frames"
            }
        )
        self.client_memory.add(
            content = asr_text,
            container_tags = ["Audio_Captions"],
            metadata = {
                "note_id": "Retrieved Audio"
            }
        )
        context_frame = self.client_memory.search.documents(
            q= query,
            container_tags = ["Frame_Captions"],
            limit = 10
        )
        context_audio = self.client_memory.search.documents(
            q= query,
            container_tags = ["Audio_Captions"],
            limit = 10
        )
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template = LLM_QUERY_PROMPT,
            input_variables = ["context_frame", "context_audio","query"]
        )
        chain = prompt | self.llm | parser
        result = await chain.ainvoke({
            "context_frame": context_frame,
            "context_audio": context_audio,
            "query" : query
        })
        print(result)
        return result

    async def delete(self) -> dict:
        frames_deleted = self.client_memory.documents.delete_bulk(container_tags=["Frame_Captions"])
        audio_deleted = self.client_memory.documents.delete_bulk(container_tags=["Audio_Captions"])
        return {"frames_deleted": frames_deleted,
                "audio_deleted": audio_deleted}
