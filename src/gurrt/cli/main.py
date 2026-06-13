import os
import logging
import sys
import time
import zipfile
import subprocess
import shutil
import urllib
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


logging.disable(logging.WARNING)

import typer
from pathlib import Path
from platformdirs import user_config_dir
import json
import asyncio

from gurrt.core.pipeline import VideoRag
from gurrt.config.config import LlamaServerManager
from gurrt.utils.llama_server_utils import download_gemma3_models
from rich.theme import Theme
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from gurrt.core.pipeline import VideoRag
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
llama_server_manager= LlamaServerManager()

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
    hf_token_link = "https://huggingface.co/settings/tokens"
    hf_token_guide = "https://huggingface.co/docs/hub/en/security-tokens"
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

    console.print(
    Panel(
        f"[info]Get your Hugging Face token here:[/info]\n"
        f"[bold green]{hf_token_link}[/bold green]\n\n"
        f"[info]Learn about authentication here:[/info]\n"
        f"[bold green]{hf_token_guide}[/bold green]",
        title="Hugging Face",
        border_style="green"
    )
)
    hf_token = Prompt.ask("[info]Enter HuggingFace Token[/info]", password=True)
    
    with open(config_file, "w") as f:
        json.dump({
            "GROQ_API_KEY": groq,
            "SUPERMEMORY_API_KEY": supermemory, 
            "HuggingFace_Token": hf_token
        }, f, indent= 2)
        
    console.print(
        Panel(
        "[success]✔ Configuration saved successfully![/success]"
        f"[success]saved at {config_file} [/success]",
        border_style= "green"
        ))

@app.command()
def init_llama():
    if not llama_server_manager.llm_path.exists() or not llama_server_manager.mmproj_path.exists():
        console.print("[error]❌ Error: Fixed GGUF model components missing from the root /models/ folder.[/error]")
        console.print("[warning]Please run this setup downloader command first:[/warning]\n👉 [bold cyan]gurrt models-download[/bold cyan]\n")
        print("gemma3 models download started")
        download_gemma3_models(llama_server_manager.models_dir) 
        #raise typer.Exit(code=1)
    if llama_server_manager.server_bin.exists():
        console.print("[success]✔ Isolated runtime server binary verified.[/success]")
        return    
    llama_server_manager.bin_dir.mkdir(parents=True, exist_ok=True)   
    console.print("[warning]⚠️ Runtime dependencies missing. Fetching latest release assets via GitHub API...[/warning]")
    try:
        req = urllib.request.Request(llama_server_manager.llama_release_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            release_data = json.loads(response.read().decode())
        
        download_url = None
        for asset in release_data.get("assets", []):
            name = asset.get("name", "").lower()
            if "bin-win" in name and "cuda" in name and "cudart" not in name and name.endswith(".zip"):
                download_url = asset.get("browser_download_url")
                break
                
        if not download_url:
            for asset in release_data.get("assets", []):
                name = asset.get("name", "").lower()
                if "bin-win" in name and "cpu" in name and name.endswith(".zip"):
                    download_url = asset.get("browser_download_url")
                    break
        
        zip_path = config_dir/ "temp_server.zip"
        console.print(f"[info]📥 Downloading dynamic release: {download_url.split('/')[-1]}...[/info]")
        
        req_dl = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_dl) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            archive_files = zip_ref.namelist()
            extracted_count = 0
            for file_path in archive_files:
                filename = os.path.basename(file_path)
                lowered_filename = filename.lower()
                if not filename:
                    continue
                if lowered_filename in ["llama-server.exe", "llama-server"]:
                    member_data = zip_ref.read(file_path)
                    with open(llama_server_manager.server_bin, "wb") as target_file:
                        target_file.write(member_data)
                    extracted_count += 1                
                elif lowered_filename.endswith(".dll"):
                    member_data = zip_ref.read(file_path)
                    dll_target_path = llama_server_manager.bin_dir / filename  # Direct to bin/
                    with open(dll_target_path, "wb") as target_file:
                        target_file.write(member_data)
                    extracted_count += 1
            
            if extracted_count == 0:
                raise FileNotFoundError("Could not locate execution components inside the release archive.")

       
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        console.print("[success]✔ Server engine and entire dependency layout successfully isolated inside /bin/![/success]")
        
    except Exception as e:
        console.print(f"[error]❌ Automation failed to retrieve or extract server asset: {e}[/error]")
        if 'zip_path' in locals() and os.path.exists(zip_path):
            os.remove(zip_path)
        raise typer.Exit(code=1)
    
@app.command()
def index_llama(video_path):
    """
    Index a video using local llama-server for captioning and embedding.
    """
    console.print(
        Panel(
            f"[primary]Indexing Video with Local Llama-Server[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    
    video_time_start = time.time()
    rag.index_video_llama_server(video_path=video_path, server_bin=llama_server_manager.server_bin, models_dir=llama_server_manager.models_dir)
    video_time_end = time.time()
    
    audio_time_start = time.time()
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    audio_time_end = time.time()
    
    console.print(Panel(
            "[success]✔ Video indexed successfully using local Llama-Server![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))
    print(f"Video Indexing Time: {video_time_end - video_time_start:.2f} seconds")
    print(f"Audio Indexing Time: {audio_time_end - audio_time_start:.2f} seconds")


@app.command()
def models_download():
    """
    Download and cache all required AI models locally.
    """
    cache_dir = config_dir/ "models"
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
    console.print(f"[success]✔ Models cached successfully at {cache_dir}![/success]")



@app.command()
def index(video_path: Path, model_name:str):
    """
    Index a video by extracting frames and audio for retrieval.
    """
    if not video_path.exists():
        console.print(
        Panel(
            f"[primary]Path Does Not Exist[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
        return
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True)
    if int(result.stdout.strip()) > 4500:
        FLAG = True
    else:
        FLAG = False
    console.print(
        Panel(
            f"[primary]Indexing Video[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    
    video_time_start = time.time()
    if model_name.lower() == ("smolvlm"):
        rag.index_video(video_path=video_path,
                        flag = FLAG)
    elif model_name.lower() == "blip2":
        rag.index_video_blip(video_path= video_path)
    video_time_end = time.time()
    
    audio_time_start = time.time()
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    audio_time_end = time.time()
    
    console.print(Panel(
            "[success]✔ Video indexed successfully![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))
    print(f"Video Indexing Time: {video_time_end - video_time_start:.2f} seconds")
    print(f"Audio Indexing Time: {audio_time_end - audio_time_start:.2f} seconds")

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