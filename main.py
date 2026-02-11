from typing import Annotated
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from app.llm import query_llm
from scripts.video_embedding_pipeline import scene_detection_frame_sampling
from pydantic import Field
import subprocess

app = FastAPI(title= "Video-Amigo")

app.get('/save_models')
async def save_models() -> JSONResponse:
    
    try:
        
        result = subprocess.run(["python", r"scripts\model_storage.py"],
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

app.get('/upload_video')
async def video_save_caption_emb(video: Annotated[str, Field(description= "Recieve Video Path")]) -> JSONResponse:
    try:
        result_frame = subprocess.run(["python", r"scripts\video_embedding_pipeline.py"],
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
        result_audio = subprocess.run(["python", r"scripts\asr_pipeline.py"],
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
    
async def chat(query: str) -> str:
    answer = await query_llm(query= query)
    return answer