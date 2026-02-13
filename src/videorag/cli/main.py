import typer
from pathlib import Path
from platformdirs import user_config_dir
import json
import asyncio
from videorag.core.pipeline import VideoRag

app = typer.Typer()
config_dir = Path(user_config_dir("videorag"))
config_dir.mkdir(exist_ok= True, parents= True)
@app.command()
def init():
    
    config_file = config_dir / "config.json"
    
    groq = typer.prompt("Enter the Groq API Key: ", hide_input= True)
    supermemory = typer.prompt("Enter the Supermemory API key: ", hide_input= True)
    ollama = typer.prompt("Enter Ollama API Key: ", hide_input= True)
    
    with open(config_file, "w") as f:
        json.dump({
            "GROQ_API_KEY": groq,
            "SUPERMEMORY_API_KEY": supermemory,
            "OLLAMA_API_KEY": ollama
        }, f, indent= 2)
    
@app.command()
def models_download():
    cache_dir = config_dir /"models"
    cache_dir.mkdir(exist_ok= True, parents= True)

    
    typer.echo("Downloading models...")
    from videorag.core.models import download_models
    download_models(cache_dir)
    

    typer.echo("Models cached successfully!")


@app.command()
def index(video_path):
    rag = VideoRag()
    rag.index_video(video_path=video_path)
    rag.index_audio(video_path=video_path)
    
@app.command()
def ask(query:str):
    rag = VideoRag()
    response = asyncio.run(rag.ask(query= query))
    typer.echo(response)
    
    
if __name__ == "__main__":
    app()