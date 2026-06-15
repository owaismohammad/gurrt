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

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory

console = ui.console

app = typer.Typer(help="gUrrT: A Video Understanding Tool")
config_dir = Path(user_config_dir("gurrt"))
config_dir.mkdir(exist_ok=True, parents=True)
llama_server_manager = LlamaServerManager()

_VALID_MODELS = {"smolvlm", "blip2"}
_session_file = config_dir / "session.json"


# ── Session persistence ───────────────────────────────────────────────────────
# spnner dots added 
# /clear command added
# added ollama in session.json and error handling it if ollama is not installed 
# added further try catches to save the session from crashing in video indexing and audio indexing
# added vram error check in init-llama, index-llama, index 
def _save_session(video_path: str) -> None:
    data: dict = {}
    if _session_file.exists():
        try:
            with open(_session_file) as f:
                data = json.load(f)
        except Exception:
            pass
    data["last_video"] = video_path
    with open(_session_file, "w") as f:
        json.dump(data, f)


def _save_ollama_flag(has_ollama: bool) -> None:
    data: dict = {}
    if _session_file.exists():
        try:
            with open(_session_file) as f:
                data = json.load(f)
        except Exception:
            pass
    data["ollama"] = has_ollama
    with open(_session_file, "w") as f:
        json.dump(data, f)


def _get_ollama_flag() -> Optional[bool]:
    if not _session_file.exists():
        return None
    try:
        with open(_session_file) as f:
            data = json.load(f)
        val = data.get("ollama")
        return bool(val) if val is not None else None
    except Exception:
        return None


def _save_gpu_info(gpu_mb: int) -> None:
    data: dict = {}
    if _session_file.exists():
        try:
            with open(_session_file) as f:
                data = json.load(f)
        except Exception:
            pass
    data["gpu_mb"] = gpu_mb
    with open(_session_file, "w") as f:
        json.dump(data, f)


def _get_gpu_mb() -> Optional[int]:
    """Returns saved GPU VRAM in MB, or None if /init hasn't been run yet."""
    if not _session_file.exists():
        return None
    try:
        with open(_session_file) as f:
            data = json.load(f)
        val = data.get("gpu_mb")
        return int(val) if val is not None else None
    except Exception:
        return None


def _detect_and_save_gpu() -> int:
    """Run nvidia-smi, save VRAM to session.json. Returns MB (0 = no GPU)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        gpu_mb = int(result.stdout.strip())
    except Exception:
        gpu_mb = 0
    _save_gpu_info(gpu_mb)
    return gpu_mb


def _save_init_done() -> None:
    data: dict = {}
    if _session_file.exists():
        try:
            with open(_session_file) as f:
                data = json.load(f)
        except Exception:
            pass
    data["init_done"] = True
    with open(_session_file, "w") as f:
        json.dump(data, f)


def _save_models_done() -> None:
    data: dict = {}
    if _session_file.exists():
        try:
            with open(_session_file) as f:
                data = json.load(f)
        except Exception:
            pass
    data["models_downloaded"] = True
    with open(_session_file, "w") as f:
        json.dump(data, f)


def _check_prereqs(command: str) -> bool:
    """Return True if /init and /models-download have both been completed.
    Prints a yellow panel listing exactly which step(s) are missing."""
    init_done = False
    models_done = False
    if _session_file.exists():
        try:
            with open(_session_file) as f:
                data = json.load(f)
            init_done = bool(data.get("init_done", False))
            models_done = bool(data.get("models_downloaded", False))
        except Exception:
            pass

    if init_done and models_done:
        return True

    missing = []
    if not init_done:
        missing.append("  [primary]/init[/primary]              — save API keys & detect GPU")
    if not models_done:
        missing.append("  [primary]/models-download[/primary]   — download AI models to disk")

    console.print(Panel(
        f"[warning]Cannot run [/warning][primary]{command}[/primary][warning] — setup incomplete.[/warning]\n\n"
        "[info]Please run the following step(s) first:[/info]\n\n" +
        "\n".join(missing),
        title="[warning]Setup Required[/warning]",
        border_style=ui.BORDER_WARNING,
    ))
    return False


def _load_session() -> tuple[Optional[VideoRag], Optional[str]]:
    if not _session_file.exists():
        return None, None
    try:
        with open(_session_file) as f:
            data = json.load(f)
        last_video = data.get("last_video", "")
        if last_video:
            rag = VideoRag()
            return rag, last_video
    except Exception:
        pass
    return None, None


# ── Slash command completion ──────────────────────────────────────────────────

_SLASH_COMMANDS = [
    ("init",            "Save API keys (Groq + Supermemory)"),
    ("init-llama",      "Download Gemma 3 + llama-server binary"),
    ("models-download", "Download all AI models locally"),
    ("index",           "Index with smolvlm or blip2              /index <path> <model>"),
    ("index-llama",     "Index with local Gemma 3  [4 GB+]        /index-llama <path>"),
    ("index-ollama",    "Index with an Ollama model               /index-ollama <path> <model>"),
    ("clear",           "Clear the screen and redraw header"),
    ("help",            "Show commands & VRAM guide"),
    ("exit",            "End session"),
]


class _SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        typed = text[1:].lower()
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(typed):
                yield Completion(
                    "/" + cmd,
                    start_position=-len(text),
                    display=f"/{cmd}",
                    display_meta=desc,
                )


# ── Help ──────────────────────────────────────────────────────────────────────

def _show_quick_start() -> None:
    console.print(Rule("[primary]Getting Started[/primary]", style=ui.BORDER_PRIMARY))
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
        console.print(f"  [primary]Step {num}[/primary]  [primary]{cmd}[/primary]")
        console.print(f"           [dim]{desc}[/dim]")
        console.print(f"           [info]{note}[/info]")
        console.print()

    console.print(Rule("[primary]Pick Your Index Command — VRAM Guide[/primary]", style=ui.BORDER_PRIMARY))
    console.print()

    vram_table = Table(
        show_header=True,
        header_style=ui.STYLE_HEADER,
        border_style=ui.BORDER_PRIMARY,
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
        border_style=ui.BORDER_PRIMARY,
    ))
    groq = Prompt.ask("[info]Groq API Key[/info]", password=True)

    console.print(Panel(
        f"[info]Get your Supermemory API Key at:[/info]\n[primary]{supermemory_link}[/primary]",
        title="[primary]Supermemory[/primary]",
        border_style=ui.BORDER_PRIMARY,
    ))
    supermemory = Prompt.ask("[info]Supermemory API Key[/info]", password=True)

    ollama_link = "https://ollama.com"
    console.print(Panel(
        f"[info]Ollama lets you run vision models locally (used by [/info][primary]/index-ollama[/primary][info]).[/info]\n"
        f"[info]Download and install it from:[/info]\n[primary]{ollama_link}[/primary]",
        title="[primary]Ollama[/primary]",
        border_style=ui.BORDER_PRIMARY,
    ))
    ollama_ans = Prompt.ask("[info]Is Ollama installed on your device?[/info]", choices=["yes", "no"], default="no")
    has_ollama = ollama_ans.lower() == "yes"
    try:
        _save_ollama_flag(has_ollama)
        with open(config_file, "w") as f:
            json.dump({"GROQ_API_KEY": groq, "SUPERMEMORY_API_KEY": supermemory}, f, indent=2)
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Failed to Save Configuration[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return

    with console.status("[info]Detecting GPU...[/info]", spinner="dots"):
        gpu_mb = _detect_and_save_gpu()
    _save_init_done()

    if gpu_mb >= 4500:
        gpu_line = f"\n[dim]GPU: {gpu_mb} MB VRAM — llama-server supported[/dim]"
    elif gpu_mb > 0:
        gpu_line = f"\n[dim]GPU: {gpu_mb} MB VRAM [/dim]"
    else:
        gpu_line = "\n[dim]GPU: not detected — CPU only[/dim]"

    console.print(Panel(
        f"[success]Configuration saved.[/success]\n[dim]{config_file}[/dim]{gpu_line}",
        border_style=ui.BORDER_SUCCESS,
    ))


def _do_init_llama() -> None:
    if not _check_prereqs("/init-llama"):
        return

    gpu_mb = _get_gpu_mb()
    if gpu_mb is None:
        console.print(Panel(
            "[warning]GPU info not available.[/warning]\n\n"
            "[dim]Run [/dim][primary]/init[/primary][dim] first — it detects your GPU automatically.[/dim]",
            title="[warning]Run /init First[/warning]",
            border_style=ui.BORDER_WARNING,
        ))
        return
    if gpu_mb < 4500:
        console.print(Panel(
            f"[error]{'No GPU detected' if gpu_mb == 0 else f'{gpu_mb} MB VRAM is not enough'}.[/error]\n\n"
            "[dim]Gemma 3 via llama-server requires 4 GB+ (4500 MB+) VRAM.[/dim]\n\n"
            "[info]Use instead:[/info]\n"
            "  [primary]/index <path> smolvlm[/primary]  — works with 4 GB VRAM\n"
            "  [primary]/index <path> blip2[/primary]    — lightest option",
            title="[error]Insufficient VRAM[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return

    if not llama_server_manager.llm_path.exists() or not llama_server_manager.mmproj_path.exists():
        try:
            with console.status("[info]Downloading Gemma 3 model weights...[/info]", spinner="dots"):
                download_gemma3_models(llama_server_manager.models_dir)
        except Exception as e:
            console.print(Panel(
                f"[error]{e}[/error]",
                title="[error]Model Download Failed[/error]",
                border_style=ui.BORDER_ERROR,
            ))
            return

    if llama_server_manager.server_bin.exists():
        ui.success("Llama server binary already present.")
        return

    llama_server_manager.bin_dir.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            llama_server_manager.llama_release_url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with console.status("[info]Fetching latest llama-server release from GitHub...[/info]", spinner="dots"):
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
        filename = download_url.split('/')[-1]

        req_dl = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with console.status(f"[info]Downloading {filename}...[/info]", spinner="dots"):
            with urllib.request.urlopen(req_dl) as response, open(zip_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

        with console.status("[info]Extracting server binary...[/info]", spinner="dots"):
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
        border_style=ui.BORDER_PRIMARY,
    ))
    from gurrt.core.models import download_models
    try:
        with console.status("[info]Downloading and caching models...[/info]", spinner="dots"):
            download_models(cache_dir)
        _save_models_done()
        ui.success(f"All models cached at {cache_dir}")
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Model Download Failed[/error]",
            border_style=ui.BORDER_ERROR,
        ))


def _do_index(video_path: str, model_name: str) -> Optional[VideoRag]:
    if not _check_prereqs("/index"):
        return None

    if model_name.lower() not in _VALID_MODELS:
        console.print(Panel(
            f"[error]Unknown model:[/error] [primary]{model_name}[/primary]\n\n"
            "[info]Available models:[/info]\n"
            "  [primary]smolvlm[/primary]  — best quality, needs 4 GB+ VRAM, but works decently with lower VRAM as well\n"
            "  [primary]blip2[/primary]    — lighter, lower VRAM requirement\n\n"
            "[dim]Example: /index lecture.mp4 smolvlm[/dim]",
            title="[error]Invalid Model[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    if not Path(video_path).exists():
        console.print(Panel(
            f"[error]File not found:[/error]\n[dim]{video_path}[/dim]\n\n"
            "[dim]Example: /index lecture.mp4 smolvlm[/dim]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    _gpu_mb = _get_gpu_mb() or 0
    FLAG = _gpu_mb > 4500
    gpu_display = f"{_gpu_mb} MB" if _gpu_mb > 0 else "not detected"

    vram_note = ""
    if model_name.lower() == "smolvlm":
        vram_note = (
            "\n[dim]4 GB+ VRAM detected — image splitting enabled[/dim]"
            if FLAG
            else f"\n[dim]{_gpu_mb} VRAM — image splitting disabled for SmolVLM[/dim]"
        )

    console.print(Panel(
        f"[info]Model:[/info] [primary]{model_name}[/primary]   "
        f"[info]GPU:[/info] [dim]{gpu_display}[/dim]\n"
        f"[dim]{video_path}[/dim]{vram_note}",
        title="[primary]Indexing Video[/primary]",
        border_style=ui.BORDER_PRIMARY,
    ))

    try:
        with console.status("[info]Loading model...[/info]", spinner="dots"):
            rag = VideoRag(reset=True)
        video_time_start = time.time()
        if model_name.lower() == "smolvlm":
            with console.status("[info]Indexing frames with SmolVLM...[/info]", spinner="dots"):
                rag.index_video(video_path=Path(video_path), flag=FLAG)
        elif model_name.lower() == "blip2":
            with console.status("[info]Indexing frames with BLIP-2...[/info]", spinner="dots"):
                rag.index_video_blip(video_path=Path(video_path))
        video_time_end = time.time()
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Video Indexing Failed[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    try:
        audio_time_start = time.time()
        with console.status("[info]Transcribing audio...[/info]", spinner="dots"):
            rag.index_audio(video_path=Path(video_path))
        audio_time_end = time.time()
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Audio Transcription Failed[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    console.print(Panel(
        f"[success]Video indexed successfully.[/success]\n"
        f"[dim]Video: {video_time_end - video_time_start:.1f}s  |  "
        f"Audio: {audio_time_end - audio_time_start:.1f}s[/dim]\n\n"
        "[primary]→[/primary] [dim]Type your question at the prompt.[/dim]",
        border_style=ui.BORDER_SUCCESS,
    ))
    _save_session(str(video_path))
    return rag


def _do_index_llama(video_path_str: str) -> Optional[VideoRag]:
    if not _check_prereqs("/index-llama"):
        return None

    gpu_mb = _get_gpu_mb()
    if gpu_mb is None:
        console.print(Panel(
            "[warning]GPU info not available.[/warning]\n\n"
            "[dim]Run [/dim][primary]/init[/primary][dim] first — it detects your GPU automatically.[/dim]",
            title="[warning]Run /init First[/warning]",
            border_style=ui.BORDER_WARNING,
        ))
        return None
    if gpu_mb < 4500:
        console.print(Panel(
            f"[error]{'No GPU detected' if gpu_mb == 0 else f'{gpu_mb} MB VRAM is not enough'}.[/error]\n\n"
            "[dim]Gemma 3 via llama-server requires 4 GB+ (4500 MB+) VRAM.[/dim]\n\n"
            "[info]Use instead:[/info]\n"
            "  [primary]/index <path> smolvlm[/primary]  — works with 4 GB VRAM\n"
            "  [primary]/index <path> blip2[/primary]    — lightest option",
            title="[error]Insufficient VRAM[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    if not llama_server_manager.server_bin.exists():
        console.print(Panel(
            "[error]llama-server binary not found.[/error]\n\n"
            "Run first:  [primary]/init-llama[/primary]\n\n"
            "[dim]This downloads the server binary and Gemma 3 model weights.[/dim]",
            title="[error]Missing Setup Step[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    console.print(Panel(
        f"[dim]{video_path_str}[/dim]\n\n"
        "[dim]Requires 4 GB+ VRAM  ·  local Gemma 3 via llama-server[/dim]",
        title="[primary]Indexing with Local Llama-Server[/primary]",
        border_style=ui.BORDER_PRIMARY,
    ))

    try:
        with console.status("[info]Loading model...[/info]", spinner="dots"):
            rag = VideoRag(reset=True)
        video_time_start = time.time()
        with console.status("[info]Indexing frames with Gemma 3 via llama-server...[/info]", spinner="dots"):
            rag.index_video_llama_server(
                video_path=Path(video_path_str),
                server_bin=llama_server_manager.server_bin,
                models_dir=llama_server_manager.models_dir,
            )
        video_time_end = time.time()
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Video Indexing Failed[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    try:
        audio_time_start = time.time()
        with console.status("[info]Transcribing audio...[/info]", spinner="dots"):
            rag.index_audio(video_path=Path(video_path_str))
        audio_time_end = time.time()
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Audio Transcription Failed[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    console.print(Panel(
        f"[success]Video indexed successfully.[/success]\n"
        f"[dim]Video: {video_time_end - video_time_start:.1f}s  |  "
        f"Audio: {audio_time_end - audio_time_start:.1f}s[/dim]\n\n"
        "[primary]→[/primary] [dim]Type your question at the prompt.[/dim]",
        border_style=ui.BORDER_SUCCESS,
    ))
    _save_session(video_path_str)
    return rag


def _do_index_ollama(video_path_str: str, model_name: str) -> Optional[VideoRag]:
    if not _check_prereqs("/index-ollama"):
        return None

    ollama_flag = _get_ollama_flag()
    if ollama_flag is False:
        console.print(Panel(
            "[error]Ollama is not installed on this device.[/error]\n\n"
            "[info]To use [/info][primary]/index-ollama[/primary][info], first install Ollama from:[/info]\n"
            "[primary]https://ollama.com[/primary]\n\n"
            "[dim]Then run [/dim][primary]/init[/primary][dim] again and answer [/dim][primary]yes[/primary]"
            "[dim] when asked about Ollama.[/dim]",
            title="[error]Ollama Not Available[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    console.print(Panel(
        f"[info]Model:[/info] [primary]{model_name}[/primary]\n"
        f"[dim]{video_path_str}[/dim]\n\n"
        "[dim]Make sure Ollama is running: [/dim][primary]ollama serve[/primary]",
        title="[primary]Indexing with Ollama[/primary]",
        border_style=ui.BORDER_PRIMARY,
    ))

    try:
        with console.status("[info]Loading model...[/info]", spinner="dots"):
            rag = VideoRag(reset=True)
        with console.status(f"[info]Indexing frames with {model_name}...[/info]", spinner="dots"):
            rag.index_video_ollama(video_path=Path(video_path_str), model_name=model_name)
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Ollama Error[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    try:
        with console.status("[info]Transcribing audio...[/info]", spinner="dots"):
            rag.index_audio(video_path=Path(video_path_str))
    except Exception as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Audio Transcription Failed[/error]",
            border_style=ui.BORDER_ERROR,
        ))
        return None

    console.print(Panel(
        "[success]Video indexed successfully.[/success]\n\n"
        "[primary]→[/primary] [dim]Type your question at the prompt.[/dim]",
        border_style=ui.BORDER_SUCCESS,
    ))
    _save_session(video_path_str)
    return rag


# ── Interactive REPL session ──────────────────────────────────────────────────

def _clean_path(s: str) -> str:
    return s.strip('"').strip("'")


def _run_session() -> None:
    _rag: Optional[VideoRag] = None
    _indexed = False

    # ── Restore last session ──────────────────────────────────────────────────
    _rag, _last_video = _load_session()
    _indexed = _rag is not None

    if _indexed:
        console.print(Panel(
            f"[success]Session resumed.[/success]\n"
            f"[dim]Last indexed: {_last_video}[/dim]\n\n"
            "[primary]→[/primary] [dim]Type your question to continue, or /help for commands.[/dim]",
            title="[primary]gUrrT Session[/primary]",
            border_style=ui.BORDER_PRIMARY,
        ))
    else:
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
            border_style=ui.BORDER_PRIMARY,
        ))

    _pt_session: PromptSession = PromptSession(
        completer=_SlashCompleter(),
        complete_while_typing=True,
        history=InMemoryHistory(),
        style=ui.pt_style,
        lexer=ui.SlashCommandLexer(),
    )

    while True:
        try:
            raw = _pt_session.prompt(ui.get_prompt_tokens(_indexed))
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
            try:
                with console.status("[info]Thinking...[/info]", spinner="dots"):
                    response = asyncio.run(_rag.ask(query=raw))
                console.print(Panel(
                    Markdown(response),
                    title="[primary]Answer[/primary]",
                    border_style=ui.BORDER_PRIMARY,
                ))
            except Exception as e:
                console.print(Panel(
                    f"[error]{e}[/error]",
                    title="[error]Query Failed[/error]",
                    border_style=ui.BORDER_ERROR,
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
                    border_style=ui.BORDER_ERROR,
                ))
                continue
            video_path = _clean_path(tokens[0])
            result = _do_index(Path(video_path), tokens[1])
            if result is not None:
                _rag, _indexed, _last_video = result, True, video_path

        elif cmd == "index-llama":
            if not rest:
                console.print(Panel(
                    "[error]Missing video path.[/error]\n\n"
                    "Usage:  [primary]/index-llama <video_path>[/primary]\n\n"
                    "[dim]Example: /index-llama lecture.mp4[/dim]",
                    border_style=ui.BORDER_ERROR,
                ))
                continue
            video_path = _clean_path(rest)
            result = _do_index_llama(video_path)
            if result is not None:
                _rag, _indexed, _last_video = result, True, video_path

        elif cmd == "index-ollama":
            tokens = rest.rsplit(maxsplit=1)
            if len(tokens) < 2:
                console.print(Panel(
                    "[error]Missing arguments.[/error]\n\n"
                    "Usage:  [primary]/index-ollama <video_path> <model_name>[/primary]\n\n"
                    "[dim]Example: /index-ollama lecture.mp4 llava[/dim]",
                    border_style=ui.BORDER_ERROR,
                ))
                continue
            video_path = _clean_path(tokens[0])
            result = _do_index_ollama(video_path, tokens[1])
            if result is not None:
                _rag, _indexed, _last_video = result, True, video_path

        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            ui.show_banner()
            if _indexed:
                console.print(Panel(
                    f"[success]Session active.[/success]\n"
                    f"[dim]Indexed: {_last_video}[/dim]\n\n"
                    "[primary]→[/primary] [dim]Type your question to continue, or /help for commands.[/dim]",
                    title="[primary]gUrrT Session[/primary]",
                    border_style=ui.BORDER_PRIMARY,
                ))
            else:
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
                    border_style=ui.BORDER_PRIMARY,
                ))

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
            try:
                with console.status("[info]Thinking...[/info]", spinner="dots"):
                    response = asyncio.run(_rag.ask(query=rest))
                console.print(Panel(
                    Markdown(response),
                    title="[primary]Answer[/primary]",
                    border_style=ui.BORDER_PRIMARY,
                ))
            except Exception as e:
                console.print(Panel(
                    f"[error]{e}[/error]",
                    title="[error]Query Failed[/error]",
                    border_style=ui.BORDER_ERROR,
                ))

        else:
            ui.warn(
                f"Unknown command: [primary]/{cmd}[/primary]  —  "
                "type [primary]/help[/primary] to see available commands."
            )


# ── Typer app (shell interface) ───────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        os.system("cls" if os.name == "nt" else "clear")
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
        border_style=ui.BORDER_PRIMARY,
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
            border_style=ui.BORDER_SUCCESS,
        ))


if __name__ == "__main__":
    app()
