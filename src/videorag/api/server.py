from typing import Annotated
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from videorag.core.llm import query_llm, delete
from pydantic import Field
import subprocess

app = FastAPI(title= "Video-Amigo")

@app.get('/save_models')
async def save_models() -> JSONResponse:
    try:
        result = subprocess.run(["python", r"core\models.py"],
                                capture_output= True,
                                check= True,
                                text= True)
        return JSONResponse(status_code=status.HTTP_200_OK, 
                            content={"message": "Resource Saved Successfully!",
                                     "output": result.stdout} )
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Script execution failed",
                "error": e.stderr
            }
        )

@app.get('/create_vectordb')
async def vectordb_creation() -> JSONResponse:
    try:
        vector_db_creation = subprocess.run(["python", r"core\vectordb.py"],
                                capture_output= True,
                                check= True,
                                text= True)
        
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Script execution failed",
                "error": e.stderr
            }
        )
    return JSONResponse(status_code=status.HTTP_200_OK, 
                            content={"message": "Resource Saved Successfully!",
                                     "frame_output": vector_db_creation.stdout})

@app.get('/upload_video')
async def video_save_caption_emb(video: Annotated[str, Field(description= "Recieve Video Path")]) -> JSONResponse:
    try:
        result_frame = subprocess.run(["python", r"core\embedding.py"],
                                capture_output= True,
                                check= True,
                                text= True)
        
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Script execution failed",
                "error": e.stderr
            }
        )
    
    try:
        result_audio = subprocess.run(["python", r"core\asr.py"],
                                capture_output= True,
                                check= True,
                                text= True)
        
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Script execution failed",
                "error": e.stderr
            }
        )
    
    return JSONResponse(status_code=status.HTTP_200_OK, 
                            content={"message": "Resource Saved Successfully!",
                                     "frame_output": result_frame.stdout,
                                     "audio_output": result_audio.stdout} )

@app.get("/llm_chat")
async def chat(query: str) -> str:
    answer = await query_llm(query= query)
    return answer

@app.get("/delete_chat")
async def delete_chat() -> JSONResponse:
    try: 
        delete()
        return JSONResponse(status_code=status.HTTP_200_OK, 
                            content={"message": "Memory Deleted Successfully!"} )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Script execution failed",
                "error": e
            }
        )