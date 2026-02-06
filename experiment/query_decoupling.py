from ollama import chat
from pydantic import BaseModel

class QueryDecouplingResponse(BaseModel):
  ASR:str
  DET:list[str]
  TYPE:list[str]



response = chat(
  model='llama3.2',
  messages=[{'role': 'system', 'content': '''decouple the user input P into 
              3 request : (i) Rasr: Requests about automatic speech recognition, to extractaudio information from the video that may pertain to the query.
              (ii) Rdet: Requests for identifyingphysical entities within the video that may assist in answering the query. 
             (iii) Rtype: Requestsfor details about the location, quantity, and relationships of the identified physical entities. These requests, which may be NULL  
             example Query: According to the video, when Mike is ready to go to the library, how many books are still placed on the square table in the middle of the room
             response:
           ```JSON
            request = {
            "ASR": "Mike is ready to go to the library",
            "DET": ["book", "table"],
            "TYPE": ["num", "loc"]
}
```  '''},{'role': 'user', 'content': 'According to the video, when the teacher leaves the classroom, how many students are still seated at their desks?'}],
  format=QueryDecouplingResponse.model_json_schema(),
)
decoupled_query_response = QueryDecouplingResponse.model_validate_json(response.message.content)
