import os
import logging
import time
import zipfile
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.disable(logging.WARNING)

import typer
from platformdirs import user_config_dir
import json
import asyncio

from gurrt.core.pipeline import VideoRag
from gurrt.config.config import LlamaServerManager
from gurrt.utils.llama_server_utils import download_gemma3_models
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.rule import Rule
from rich.markdown import Markdown
from gurrt.cli import ui

console = ui.console

app = typer.Typer(help="gUrrT: A Video Understanding Tool")
config_dir = Path(user_config_dir("gurrt"))
config_dir.mkdir(exist_ok=True, parents=True)
llama_server_manager = LlamaServerManager()

_VALID_MODELS = {"smolvlm", "blip2"}


# ── Help ──────────────────────────────────────────────────────────────────────

def _show_quick_start() -> None:
    console.print(Rule("[primary]Getting Started[/primary]", style="cyan"))
    console.print()

    workflow = [
        (
            "1",
            "/init",
            "Save your API keys (Groq + Supermemory)",
            "Run once — required for cloud LLM inference",
        ),
        (
            "2",
            "/models-download",
            "Download and cache all AI models locally",
            "Run once before indexing for the first time",
        ),
        (
            "3",
            "/index <path> <model>  or  /index-llama <path>  or  /index-ollama <path> <model>",
            "Index your video",
            "Pick the command that matches your VRAM — see guide below",
        ),
        (
            "4",
            "type your question",
            "Ask anything about the indexed video",
            "No slash needed — type directly at the prompt",
        ),
    ]

    for num, cmd, desc, note in workflow:
        console.print(f"  [primary]Step {num}[/primary]  [bold bright_cyan]{cmd}[/bold bright_cyan]")
        console.print(f"           [dim]{desc}[/dim]")
        console.print(f"           [info]{note}[/info]")
        console.print()

    console.print(Rule("[primary]Pick Your Index Command — VRAM Guide[/primary]", style="cyan"))
    console.print()

    vram_table = Table(
        show_header=True,
        header_style="bold bright_cyan",
        border_style="cyan",
        show_lines=True,
    )
    vram_table.add_column("Your VRAM", style="primary", no_wrap=True, min_width=10)
    vram_table.add_column("Command", min_width=44)
    vram_table.add_column("Notes", style="dim", min_width=34)

    vram_table.add_row(
        "4 GB",
        "/index video.mp4 [primary]smolvlm[/primary]\n"
        "/index video.mp4 [primary]blip2[/primary]\n"
        "/index-ollama video.mp4 [primary]<model>[/primary]",
        "smolvlm — best quality in 4 GB\n"
        "blip2   — lightest option\n"
        "Ollama  — use a ≤4 GB quant\n"
        "          e.g. llava:7b-q4_K_M",
    )
    vram_table.add_row(
        "4 GB +",
        "/index video.mp4 [primary]smolvlm[/primary]\n"
        "/index video.mp4 [primary]blip2[/primary]\n"
        "/index-ollama video.mp4 [primary]<model>[/primary]\n"
        "[primary]/index-llama[/primary] video.mp4",
        "All 4 GB options, plus:\n"
        "index-llama uses Gemma 3 locally\n"
        "via llama-server — higher quality\n"
        "→ run [primary]/init-llama[/primary] first",
    )
    console.print(vram_table)
    console.print()
    console.print(
        "[dim]Commands also work from the shell as [/dim][primary]gurrt <command>[/primary]"
        "[dim] — run [/dim][primary]gurrt <command> --help[/primary][dim] for details.[/dim]"
    )


# ── Shared command logic ──────────────────────────────────────────────────────

def _do_init() -> None:
    groq_link = "https://console.groq.com/docs/models"
    supermemory_link = "https://supermemory.ai/docs/integrations/supermemory-sdk"
    config_file = config_dir / "config.json"

    console.print(Panel(
        f"[info]Get your Groq API Key at:[/info]\n[primary]{groq_link}[/primary]",
        title="[primary]Groq[/primary]",
        border_style="cyan",
    ))
    groq = Prompt.ask("[info]Groq API Key[/info]", password=True)

    console.print(Panel(
        f"[info]Get your Supermemory API Key at:[/info]\n[primary]{supermemory_link}[/primary]",
        title="[primary]Supermemory[/primary]",
        border_style="cyan",
    ))
    supermemory = Prompt.ask("[info]Supermemory API Key[/info]", password=True)

    with open(config_file, "w") as f:
        json.dump({"GROQ_API_KEY": groq, "SUPERMEMORY_API_KEY": supermemory}, f, indent=2)

    console.print(Panel(
        f"[success]Configuration saved.[/success]\n[dim]{config_file}[/dim]",
        border_style="bright_green",
    ))


def _do_init_llama() -> None:
    if not llama_server_manager.llm_path.exists() or not llama_server_manager.mmproj_path.exists():
        ui.step("Downloading Gemma 3 model weights...")
        download_gemma3_models(llama_server_manager.models_dir)

    if llama_server_manager.server_bin.exists():
        ui.success("Llama server binary already present.")
        return

    llama_server_manager.bin_dir.mkdir(parents=True, exist_ok=True)
    ui.step("Fetching latest llama-server release from GitHub...")
    try:
        req = urllib.request.Request(
            llama_server_manager.llama_release_url, headers={"User-Agent": "Mozilla/5.0"}
        )
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

        zip_path = config_dir / "temp_server.zip"
        ui.step(f"Downloading {download_url.split('/')[-1]}...")

        req_dl = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_dl) as response, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            extracted_count = 0
            for file_path in zip_ref.namelist():
                filename = os.path.basename(file_path)
                lowered = filename.lower()
                if not filename:
                    continue
                if lowered in ["llama-server.exe", "llama-server"]:
                    with open(llama_server_manager.server_bin, "wb") as f:
                        f.write(zip_ref.read(file_path))
                    extracted_count += 1
                elif lowered.endswith(".dll"):
                    with open(llama_server_manager.bin_dir / filename, "wb") as f:
                        f.write(zip_ref.read(file_path))
                    extracted_count += 1

            if extracted_count == 0:
                raise FileNotFoundError("Could not locate execution components inside the release archive.")

        if os.path.exists(zip_path):
            os.remove(zip_path)

        ui.success("Server binary and dependencies installed.")

    except Exception as e:
        ui.error(f"Failed to download or extract server: {e}")
        if "zip_path" in locals() and os.path.exists(zip_path):
            os.remove(zip_path)


def _do_models_download() -> None:
    cache_dir = config_dir / "models"
    cache_dir.mkdir(exist_ok=True, parents=True)

    console.print(Panel(
        f"[dim]Cache location: {cache_dir}[/dim]",
        title="[primary]Downloading Models[/primary]",
        border_style="cyan",
    ))
    from gurrt.core.models import download_models
    download_models(cache_dir)
    ui.success(f"All models cached at {cache_dir}")


def _do_index(video_path: str, model_name: str) -> Optional[VideoRag]:
    if model_name.lower() not in _VALID_MODELS:
        console.print(Panel(
            f"[error]Unknown model:[/error] [primary]{model_name}[/primary]\n\n"
            "[info]Available models:[/info]\n"
            "  [primary]smolvlm[/primary]  — best quality, needs 4 GB+ VRAM\n"
            "  [primary]blip2[/primary]    — lighter, lower VRAM requirement\n\n"
            "[dim]Example: /index lecture.mp4 smolvlm[/dim]",
            title="[error]Invalid Model[/error]",
            border_style="bright_red",
        ))
        return None

    if not Path(video_path).exists():
        console.print(Panel(
            f"[error]File not found:[/error]\n[dim]{video_path}[/dim]\n\n"
            "[dim]Example: /index lecture.mp4 smolvlm[/dim]",
            border_style="bright_red",
        ))
        return None

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    gpu_mb = result.stdout.strip()
    FLAG = int(gpu_mb) > 4500

    vram_note = ""
    if model_name.lower() == "smolvlm":
        vram_note = (
            "\n[dim]4 GB+ VRAM detected — image splitting enabled[/dim]"
            if FLAG
            else "\n[dim]< 4 GB VRAM — image splitting disabled for SmolVLM[/dim]"
        )

    console.print(Panel(
        f"[info]Model:[/info] [primary]{model_name}[/primary]   "
        f"[info]GPU:[/info] [dim]{gpu_mb} MB[/dim]\n"
        f"[dim]{video_path}[/dim]{vram_note}",
        title="[primary]Indexing Video[/primary]",
        border_style="cyan",
    ))

    rag = VideoRag(reset=True)
    video_time_start = time.time()
    if model_name.lower() == "smolvlm":
        rag.index_video(video_path=Path(video_path), flag=FLAG)
    elif model_name.lower() == "blip2":
        rag.index_video_blip(video_path=Path(video_path))
    video_time_end = time.time()

    audio_time_start = time.time()
    with console.status("[info]Transcribing audio...[/info]", spinner="dots"):
        rag.index_audio(video_path=Path(video_path))
    audio_time_end = time.time()

    console.print(Panel(
        f"[success]Video indexed successfully.[/success]\n"
        f"[dim]Video: {video_time_end - video_time_start:.1f}s  |  "
        f"Audio: {audio_time_end - audio_time_start:.1f}s[/dim]\n\n"
        "[primary]→[/primary] [dim]Type your question at the prompt.[/dim]",
        border_style="bright_green",
    ))
    return rag


def _do_index_llama(video_path_str: str) -> Optional[VideoRag]:
    if not llama_server_manager.server_bin.exists():
        console.print(Panel(
            "[error]llama-server binary not found.[/error]\n\n"
            "Run first:  [primary]/init-llama[/primary]\n\n"
            "[dim]This downloads the server binary and Gemma 3 model weights.[/dim]",
            title="[error]Missing Setup Step[/error]",
            border_style="bright_red",
        ))
        return None

    console.print(Panel(
        f"[dim]{video_path_str}[/dim]\n\n"
        "[dim]Requires 4 GB+ VRAM  ·  local Gemma 3 via llama-server[/dim]",
        title="[primary]Indexing with Local Llama-Server[/primary]",
        border_style="cyan",
    ))

    rag = VideoRag(reset=True)
    video_time_start = time.time()
    rag.index_video_llama_server(
        video_path=Path(video_path_str),
        server_bin=llama_server_manager.server_bin,
        models_dir=llama_server_manager.models_dir,
    )
    video_time_end = time.time()

    audio_time_start = time.time()
    with console.status("[info]Transcribing audio...[/info]", spinner="dots"):
        rag.index_audio(video_path=Path(video_path_str))
    audio_time_end = time.time()

    console.print(Panel(
        f"[success]Video indexed successfully.[/success]\n"
        f"[dim]Video: {video_time_end - video_time_start:.1f}s  |  "
        f"Audio: {audio_time_end - audio_time_start:.1f}s[/dim]\n\n"
        "[primary]→[/primary] [dim]Type your question at the prompt.[/dim]",
        border_style="bright_green",
    ))
    return rag


def _do_index_ollama(video_path_str: str, model_name: str) -> Optional[VideoRag]:
    console.print(Panel(
        f"[info]Model:[/info] [primary]{model_name}[/primary]\n"
        f"[dim]{video_path_str}[/dim]\n\n"
        "[dim]Make sure Ollama is running: [/dim][primary]ollama serve[/primary]",
        title="[primary]Indexing with Ollama[/primary]",
        border_style="cyan",
    ))

    rag = VideoRag(reset=True)
    rag.index_video_ollama(video_path=Path(video_path_str), model_name=model_name)

    with console.status("[info]Transcribing audio...[/info]", spinner="dots"):
        rag.index_audio(video_path=Path(video_path_str))

    console.print(Panel(
        "[success]Video indexed successfully.[/success]\n\n"
        "[primary]→[/primary] [dim]Type your question at the prompt.[/dim]",
        border_style="bright_green",
    ))
    return rag


# ── Interactive REPL session ──────────────────────────────────────────────────

def _run_session() -> None:
    _rag: Optional[VideoRag] = None
    _indexed = False

    console.print(Panel(
        "[info]Type a question to ask about your video, or use a slash command.[/info]\n\n"
        "[dim]"
        "  /help                          show commands & VRAM guide\n"
        "  /init                          save API keys\n"
        "  /init-llama                    download Gemma 3 + server binary\n"
        "  /models-download               download all AI models\n"
        "  /index <path> <model>          index with smolvlm or blip2\n"
        "  /index-llama <path>            index with local Gemma 3  (needs /init-llama)\n"
        "  /index-ollama <path> <model>   index with an Ollama model\n"
        "  /exit                          end session"
        "[/dim]",
        title="[primary]gUrrT Session[/primary]",
        border_style="cyan",
    ))

    while True:
        status = "[success]●[/success]" if _indexed else "[dim]○[/dim]"
        try:
            raw = Prompt.ask(f"\n[primary]gurrt[/primary] {status} [primary]❯[/primary]")
        except (KeyboardInterrupt, EOFError):
            console.print()
            ui.info("Goodbye.")
            return

        raw = raw.strip()
        if not raw:
            continue

        # ── Direct question ───────────────────────────────────────────────────
        if not raw.startswith("/"):
            if not _indexed or _rag is None:
                ui.warn(
                    "No video indexed yet. "
                    "Use [primary]/index[/primary], [primary]/index-llama[/primary], "
                    "or [primary]/index-ollama[/primary] first."
                )
                continue
            with console.status("[info]Thinking...[/info]", spinner="dots"):
                response = asyncio.run(_rag.ask(query=raw))
            console.print(Panel(
                Markdown(response),
                title="[success]Answer[/success]",
                border_style="bright_green",
            ))
            continue

        # ── Slash command ─────────────────────────────────────────────────────
        parts = raw[1:].split(maxsplit=1)
        cmd  = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("exit", "quit", "q"):
            ui.info("Goodbye.")
            return

        elif cmd in ("help", "?"):
            _show_quick_start()

        elif cmd == "init":
            _do_init()

        elif cmd == "init-llama":
            _do_init_llama()

        elif cmd == "models-download":
            _do_models_download()

        elif cmd == "index":
            # rsplit so paths containing spaces work: last token = model, rest = path
            tokens = rest.rsplit(maxsplit=1)
            if len(tokens) < 2:
                console.print(Panel(
                    "[error]Missing arguments.[/error]\n\n"
                    "Usage:  [primary]/index <video_path> <model_name>[/primary]\n\n"
                    "  [primary]smolvlm[/primary]  — best quality, 4 GB+ VRAM\n"
                    "  [primary]blip2[/primary]    — lighter, lower VRAM\n\n"
                    "[dim]Example: /index lecture.mp4 smolvlm[/dim]",
                    border_style="bright_red",
                ))
                continue
            result = _do_index(Path(tokens[0]), tokens[1])
            if result is not None:
                _rag, _indexed = result, True

        elif cmd == "index-llama":
            if not rest:
                console.print(Panel(
                    "[error]Missing video path.[/error]\n\n"
                    "Usage:  [primary]/index-llama <video_path>[/primary]\n\n"
                    "[dim]Example: /index-llama lecture.mp4[/dim]",
                    border_style="bright_red",
                ))
                continue
            result = _do_index_llama(rest)
            if result is not None:
                _rag, _indexed = result, True

        elif cmd == "index-ollama":
            tokens = rest.rsplit(maxsplit=1)
            if len(tokens) < 2:
                console.print(Panel(
                    "[error]Missing arguments.[/error]\n\n"
                    "Usage:  [primary]/index-ollama <video_path> <model_name>[/primary]\n\n"
                    "[dim]Example: /index-ollama lecture.mp4 llava[/dim]",
                    border_style="bright_red",
                ))
                continue
            result = _do_index_ollama(tokens[0], tokens[1])
            if result is not None:
                _rag, _indexed = result, True

        elif cmd == "ask":
            if not _indexed or _rag is None:
                ui.warn(
                    "No video indexed yet. "
                    "Use [primary]/index[/primary], [primary]/index-llama[/primary], "
                    "or [primary]/index-ollama[/primary] first."
                )
                continue
            if not rest:
                ui.info("Type your question after /ask, or just type it directly at the prompt.")
                continue
            with console.status("[info]Thinking...[/info]", spinner="dots"):
                response = asyncio.run(_rag.ask(query=rest))
            console.print(Panel(
                Markdown(response),
                title="[success]Answer[/success]",
                border_style="bright_green",
            ))

        else:
            ui.warn(
                f"Unknown command: [primary]/{cmd}[/primary]  —  "
                "type [primary]/help[/primary] to see available commands."
            )


# ── Typer app (shell interface) ───────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    ui.show_banner()
    if ctx.invoked_subcommand is None:
        _run_session()


@app.command()
def init():
    """Save your Groq and Supermemory API keys."""
    _do_init()


@app.command()
def init_llama():
    """Download the Gemma 3 model weights and llama-server binary."""
    _do_init_llama()


@app.command()
def models_download():
    """Download and cache all required AI models locally.

    Only needed once. Models are cached in the user config directory.
    """
    _do_models_download()


@app.command()
def index(
    video_path: Path = typer.Argument(
        ...,
        help="Path to the video file to index.  Example: lecture.mp4",
        metavar="VIDEO_PATH",
    ),
    model_name: str = typer.Argument(
        ...,
        help="Vision model to use for frame captioning.  Choices: smolvlm | blip2",
        metavar="MODEL_NAME",
    ),
):
    """Index a video by extracting and captioning frames + transcribing audio.

    MODEL_NAME choices:

        smolvlm  — best caption quality, requires 4 GB+ VRAM

        blip2    — lighter model, lower VRAM requirement

    Example:

        gurrt index lecture.mp4 smolvlm
    """
    result = _do_index(video_path, model_name)
    if result is None:
        raise typer.Exit(code=1)


@app.command()
def index_llama(
    video_path: str = typer.Argument(
        ...,
        help="Path to the video file to index.  Example: lecture.mp4",
        metavar="VIDEO_PATH",
    ),
):
    """Index a video using the local Gemma 3 llama-server (no cloud API needed).

    Requires 4 GB+ VRAM. Before running this command you must first download
    the server binary and model weights:

        gurrt init-llama

    Then index your video:

        gurrt index-llama lecture.mp4
    """
    result = _do_index_llama(video_path)
    if result is None:
        raise typer.Exit(code=1)


@app.command()
def index_ollama(
    video_path: str = typer.Argument(
        ...,
        help="Path to the video file to index.  Example: lecture.mp4",
        metavar="VIDEO_PATH",
    ),
    model_name: str = typer.Argument(
        ...,
        help="Name of the Ollama vision model to use.  Example: llava, bakllava, llava-llama3",
        metavar="MODEL_NAME",
    ),
):
    """Index a video using a locally running Ollama vision model.

    Any multimodal Ollama model that accepts images can be used.
    Make sure Ollama is running and the model is already pulled.

    Example:

        ollama pull llava

        gurrt index-ollama lecture.mp4 llava
    """
    _do_index_ollama(video_path, model_name)


@app.command()
def ask():
    """Start a standalone Q&A session about the last indexed video.

    The video must be indexed first using one of the index commands.
    Type your question and press Enter. Type 'exit' or press Ctrl+C to quit.

    Example:

        gurrt ask
    """
    rag = VideoRag()
    console.print(Panel(
        "[info]Ask anything about your indexed video.[/info]\n"
        "[dim]Type [/dim][primary]exit[/primary][dim] or press "
        "[/dim][primary]Ctrl+C[/primary][dim] to end the session.[/dim]",
        title="[primary]Interactive Q&A[/primary]",
        border_style="cyan",
    ))
    while True:
        try:
            query = Prompt.ask("\n[primary]You[/primary]")
        except (KeyboardInterrupt, EOFError):
            console.print()
            ui.info("Session ended.")
            break
        query = query.strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            ui.info("Session ended.")
            break
        with console.status("[info]Thinking...[/info]", spinner="dots"):
            response = asyncio.run(rag.ask(query=query))
        console.print(Panel(
            Markdown(response),
            title="[success]Answer[/success]",
            border_style="bright_green",
        ))


if __name__ == "__main__":
    app()
