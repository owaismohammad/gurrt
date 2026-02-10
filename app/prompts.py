BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_QUERY_PROMPT = """ YOU ARE CURRENTLY ANSWERING A USERS QUERY BASED ON A VIDEO \n
below are is the relevant context that has been derived from the video that has been similiar to the prompt use this as reference \n
{frame_context}
\n
\n
BELOW IS THE CONTEXT DERIVED FROM THE TRANSCRIPT OF THE VIDEO
{asr_context}
THE QUERY ASKED BY THE USER IS GIVEN BELOW:
{query}
"""