BLIP_CUSTOM_PROMPT = "A detailed description of what is going on in this picture: "

LLM_QUERY_PROMPT = """ YOU ARE CURRENTLY ANSWERING A USERS QUERY BASED ON A VIDEO \n
below are is the relevant context that has been derived from the video that has been similiar to the prompt use this as reference \n
\n
\n

{context}
THE QUERY ASKED BY THE USER IS GIVEN BELOW:
{query}
"""