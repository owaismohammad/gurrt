import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.prompts import LLM_QUERY_PROMPT
from scripts.collections_query import query_collection

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY is None:
    raise RuntimeError("GROQ_API_KEY not found")

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    max_tokens= 4096,
)


def query_llm(query:str) -> str:
    caption_list, asr_list = query_collection(query= query , n_results= 10)
    # asr_list = query_asr_collection()
    # ocr_list = query_ocr_collection()
    
    # context_caption = "\n".join(caption_list[0])
    # asr_text = "\n".join(asr_list[0])
    # parser = StrOutputParser()
    # prompt = PromptTemplate(
    #     template = LLM_QUERY_PROMPT,
    #     input_variables = ["frame_context", "asr_context","query"]
    # )
    # chain = prompt | llm | parser
    # result = chain.invoke({
    #     "frame_context": context_caption,
    #     "asr_context": asr_text,
    #     "query" : query
    # })
    print(caption_list)
    print(asr_list)
    # return result


result = query_llm("what does the person really mean by when he is taking the limit x raise to the power of h minus 1 divided by h")
print(result)