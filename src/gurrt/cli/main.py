import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.disable(logging.WARNING)

import typer
from pathlib import Path
from platformdirs import user_config_dir
import json
import asyncio
from gurrt.core.pipeline import VideoRag

from rich.theme import Theme
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

custom_theme = Theme({
    "primary": "bold green",
    "success": "bold bright_green",
    "error": "bold red",
    "info": "green",
    "warning": "yellow"
})

console = Console(theme= custom_theme)


app = typer.Typer(help= "🌿 gUrrT: A Video Understanding Tool")

config_dir = Path(user_config_dir("gurrt"))
config_dir.mkdir(exist_ok= True, parents= True)


@app.callback()
def main():
    title = Text("🌿 gUrrT: A Video Understanding Tool", style="bold bright_green")
    console.print(Rule(title, style="green"))
@app.command()
def init():
    """
    Initialize VideoRag by saving required API keys.
    """
    groq_link = "https://console.groq.com/docs/models"
    supermemory_link = "https://supermemory.ai/docs/integrations/supermemory-sdk"
    config_file = config_dir / "config.json"
    console.print(
        Panel(
            "[info]Get your Groq API Key here:\n[/info]"
            f"[bold green]{groq_link}[/bold green]",
            title="Groq",
            border_style="green"
        )
    )
    groq = Prompt.ask("[info]Enter Groq API Key[/info]", password=True)
    
    console.print(
        Panel(
            "[info]Get your Supermemory API Key here:\n[/info]"
            f"[bold green]{supermemory_link}[/bold green]",
            title="Supermemory",
            border_style="green"
        )
    )
    supermemory = Prompt.ask("[primary]Enter Supermemory API Key[/primary]", password=True)

    
    
    with open(config_file, "w") as f:
        json.dump({
            "GROQ_API_KEY": groq,
            "SUPERMEMORY_API_KEY": supermemory,
        }, f, indent= 2)
        
    console.print(
        Panel(
        "[success]✔ Configuration saved successfully![/success]"
        f"[success]saved at {config_file} [/success]",
        border_style= "green"
        ))
        
    
@app.command()
def models_download():
    """
    Download and cache all required AI models locally.
    """
    cache_dir = config_dir /"models"
    cache_dir.mkdir(exist_ok= True, parents= True)

    console.print(
        Panel(
            "[primary]Downloading Models[/primary]",
            border_style="green"
        )
    )
    from gurrt.core.models import download_models
    with Progress(
        SpinnerColumn(style="green"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="green"),
        console=console,
    ) as progress:
        task = progress.add_task("[info]Downloading models...", total=100)
        download_models(cache_dir)
        progress.update(task, completed=100)

    console.print("[success]✔ Models cached successfully![/success]")



@app.command()
def index(video_path):
    """
    Index a video by extracting frames and audio for retrieval.
    """
    console.print(
        Panel(
            f"[primary]Indexing Video[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    rag.index_video(video_path=video_path)
    
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    
    console.print(Panel(
            "[success]✔ Video indexed successfully![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))

@app.command()
def index_ollama(video_path, model_name):
    """
    Index a video by extracting frames and audio for retrieval with Ollama Models.
    Plug in your Ollama Model Name
    """
    console.print(
        Panel(
            f"[primary]Indexing Video With Ollama[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    rag.index_video_ollama(video_path=video_path, model_name= model_name)
    
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    
    console.print(Panel(
            "[success]✔ Video indexed successfully![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))
    
@app.command(help = "Ask a question about an indexed video.")
def ask(query:str):
    """
    Ask a question about an indexed video.
    """
    rag = VideoRag()
    
    with console.status("[info]Thinking...[/info]", spinner="dots"):
        response = asyncio.run(rag.ask(query= query))
    console.print(
        Panel(
            response,
            title="[success]Answer[/success]",
            border_style="green"
        )
    )
        
if __name__ == "__main__":
    app()