import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.prompts import LLM_QUERY_PROMPT
from scripts.collections_query import query_collection

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supermemory import Supermemory
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY")
if GROQ_API_KEY is None:
    raise RuntimeError("GROQ_API_KEY not found")
elif SUPERMEMORY_API_KEY is None:
    raise RuntimeError("SUPERMEMORY_API_KEY not found")

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    max_tokens= 4096,
)
client_memory = Supermemory(
    api_key=SUPERMEMORY_API_KEY,
)

async def query_llm(query:str) -> str:
    caption_list, asr_list = query_collection(query= query , n_results= 10)

    context_caption = "\n".join(caption_list)
    asr_text = "\n".join(asr_list)
    
    client_memory.add(
        content = context_caption,
        container_tags = ["Video-Rag"],
        metadata = {
            "note_id": "Retrieved Frames"
        }
    )
    client_memory.add(
        content = context_caption,
        container_tags = ["Video-Rag"],
        metadata = {
            "note_id": "Retrieved Audio"
        }
    )
    context = client_memory.search.documents(
        q= query,
        container_tags = ["Video-Rag"],
        limit = 10
    )
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template = LLM_QUERY_PROMPT,
        input_variables = ["context","query"]
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "context": context,
        "query" : query
    })
    return result

def delete():
    client_memory.documents.delete_bulk(container_tags=["Video-Rag"])

# result = query_llm("tell me what exactly is the video talking about in 10 points az")
# print(result)
# delete()

