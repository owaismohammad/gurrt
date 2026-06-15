[![PyPI version](https://img.shields.io/pypi/v/gurrt)](https://pypi.org/project/gurrt/) [![Python Versions](https://img.shields.io/pypi/pyversions/gurrt)](https://pypi.org/project/gurrt/) 
[![License](https://img.shields.io/pypi/l/gurrt)](https://pypi.org/project/gurrt/) [![Downloads](https://pepy.tech/badge/gurrt/)](https://pepy.tech/project/gurrt) [![Twitter Follow](https://img.shields.io/twitter/follow/muffBozo.svg?style=social)](https://twitter.com/muffBozo) [![Twitter Follow](https://img.shields.io/twitter/follow/as_farrr.svg?style=social)](https://x.com/as_farrr)

<p align="center">
  <img src="https://raw.githubusercontent.com/owaismohammad/gurrt/main/gurrt.png" width="450">
</p>

<h1 align="center">gUrrT · Conversational Video Intelligence</h1>

We study a lot on YouTube and questions keep coming up mid lecture. So we wander
Google, ChatGPT, Claude and most of the time still don't get what we actually need:

- **Google** returns generic explanations with no idea what was just taught
- **Claude/Gemini 3 (Free Tier)** reason well but have never seen the video, answers from the model's generic knowledge base, not lecture informed answers
- **YouTube's Ask** is a paid Premium feature, transcript only, and blind to anything on the board
- **Gemini 3.x / GPT  (Premium) with video** process frames but require Re-uploading every session, hit duration limits on long lectures, and keep your video on their servers

The real answer is locked inside the video. The person teaching could answer instantly.
Or someone smarter who already watched the whole thing.

**gUrrT builds that person.**

## Why Existing Solutions Fall Short

Large Video Language Models (LVLMs) are the current standard for video understanding but they require **16 GB of VRAM at minimum**, with the most capable ones demanding up to
**80 GB**, running on enterprise A100s and H100s.[^1]

Open Source LVLMs that run locally process **64–256 frames total** at 1 FPS, that is
roughly **1 to 4 minutes** of video.[^2] A 1 hour or even a 30 minute lecture is out of reach. Even with
4 bit quantization, a 7B video model needs 6–8 GB just to load ruling out 4 GB hardware entirely.

Cloud models like Gemini handle up to 1 hour of video, but run on Google's
infrastructure your video leaves your machine, you depend on internet access,
and you pay per token.[^3]

Smaller video LLMs at the 4 to 8 GB range exist but share the same root problem as
the large ones: **uniform frame sampling**. At 1 FPS, a 30min lecture becomes
1,800+ frames the same static slide captured hundreds of times, feeding the LLM
redundant, noisy context regardless of model size.

- > [^1]: Qwen2-VL 7B requires ~16 GB at fp16; InternVL2-40B and 72B-class models require 80 GB+
- > [^2]: LLaVA-Video processes 64 frames by default; Qwen2-VL supports up to 256 frames
- > [^3]: Gemini 3 supports video only up to 1 hour with duration limits

## Where gUrrT Comes In

When you feed a video into an LVLM or upload it to Gemini or Claude what is actually inside that video? Frame context and audio context.
That is it. The rest is just the LLM reasoning over that context.

So what if you could build that context yourself?

LVLMs handle video as a whole the frames, the audio, the temporal dimension
but what actually goes into the LLM is just frame captions and a transcript.
All that compute, all that VRAM, all those cloud upload limits and pricing plans
just so a model can build context for an LLM to reason over.

gUrrT builds that context instead.

Smartly extract only the frames where content genuinely changed.
Transcribe the audio with Whisper. Index everything into a searchable vector store.
Then hand that context to a state of the art LLM for reasoning
no LVLM in the pipeline, no 80 GB GPU, no video leaving your machine.

The LLM does not need to understand video. It just needs the right context.
That is the problem gUrrT solves.

> gUrrT currently uses Groq for LLM inference fast and free tier accessible.
> Ollama and llama.cpp is already supported for local inference and will be the default
> in future versions for complete independence.

```
Video
 │
 ├── Frame Extraction (The Eyes)
 │     Adaptive sampling — only frames where content meaningfully changed
 │     ↓ Captioning model generates descriptions
 │     ↓ CLIP embeds each caption
 │     ↓ Stored in ChromaDB
 │
 ├── Audio Pipeline (The Ears)
 │     FFmpeg demux → Faster-Whisper transcription
 │     ↓ Chunked and embedded with CLIP
 │     ↓ Stored in ChromaDB (separate collection)
 │
 └── Query Time
       User question → CLIP embedding → dual retrieval
       CrossEncoder reranking
       LLM synthesizes answer (Groq cloud or Ollama local)
```




The quality of gUrrT's answers depends entirely on the quality of the context it builds as established [above](#where-gurrt-comes-in). v1 built bad context. v2 builds the right context, faster, with better models.

**v1 had two distinct failure modes:**

**Oversampling** — when scene detection found nothing (which was most of the time on lecture videos), the pipeline fell back to uniform 1 FPS sampling. A 1:45 minute video produced 105 frames, nearly all of them the same slide. The LLM received hundreds of identical captions and had to reason through that noise.

**Undersampling** — when scene detection did run, it was tuned for cinematic hard cuts full frame colour changes between shots. Lecture slides change by text appearing, not by scene switching. The detector found almost nothing: **2 frames in a 23-minute lecture, 5 in a 53-minute lecture, 6 in a full hour**. The LLM had near zero visual context for hour long content.

Both modes produced wrong context. One was too much garbage. The other was too little signal.

**v2's Temporal Persistence Filter resolves both** it detects genuine slide changes by tracking visual hash differences across a persistence window, filtering out speaker movement and transition artifacts. 

The results across the same lectures on 6GB VRAM, RTX4050:

| Video Length | Total Frames | v1 Key Frames | v2 Key Frames | v1 Time | v2 Time |
|-------------|-------------|--------------|--------------|---------|---------|
| 1min 45secs | 3,165 | 105 (uniform, all noise) | **7** | 48s | ~12s |
| 4min 19secs | 7,793 | 48 | **11** | 36s | ~22s |
| 13min | 46,809 | 33 | **61** | 171s | ~161s |
| 23min 46secs | 85,610 | 2 (scene detection blind) | **42** | 266s | ~153s |
| 53min 40secs | 193,218 | 5 (scene detection blind) | **147** | 598s | ~256s |
| 1hr | 107,523 | 6 (scene detection blind) | **36** | 395s | ~111s |

For short videos, v2 strips the noise 105 redundant frames down to 7 genuine slide changes. For longer lectures where scene detection was effectively blind, v2 surfaces the content that was always there but never captured. Every selected frame is an actual content change, not a duplicate or a speaker mid gesture. Speed improvement follows naturally: **1.7× to 4× faster** across the board, because the pipeline is no longer wasting time captioning frames that carry no information.

| | v1 | v2 |
|--|--|--|
| **Captioning** | BLIP2 only | SmolVLM · BLIP2 · Gemma 3 via llama.cpp · Ollama |
| **Audio** | MoviePy + Whisper large-v2 | FFmpeg demux + distil-large-v2 batched |
| **Interface** | None | TUI, slash commands, autocomplete, session persistence |

Captioning has graduated from BLIP2 to Gemma 3 4B via llama.cpp a model that can actually read text off a slide. Pick the backend that matches your hardware:

| Backend | Command | VRAM |
|---------|---------|------|
| SmolVLM 500M | `/index <path> smolvlm` | 4 GB |
| BLIP-2 | `/index <path> blip2` | 4 GB |
| Gemma 3 4B via llama.cpp | `/index-llama <path>` | 4 GB+ |
| Any Ollama vision model | `/index-ollama <path> <model>` | varies |

LLM inference runs on **Groq** (cloud, free tier) or **Llama.cpp**/**Ollama** (fully local, in future versions).

## Installation

Requires **Python 3.12**. Install PyTorch first the GPU/CPU
variant cannot be auto selected by a package manager.

### pip

```bash
# create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install PyTorch, pick one
# GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU  
pip install torch torchvision torchaudio                                                       

# install gurrt
pip install gurrt
```

### uv

```bash
pip install uv
uv init my-project && cd my-project
uv python pin 3.12
uv add gurrt
```

Add to `pyproject.toml` for CUDA PyTorch routing, then run `uv sync`:

```toml
[project]
dependencies = ["torch", "torchvision", "torchaudio"]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

---

## How to Use

Type `gurrt` and the session starts. Commands are slash prefixed type `/` for
autocomplete. Previous sessions resume automatically.

**First time — run in this order:**

```
/init  →  /models-download  →  /index <path> <model>  →  ask your question
```

### Commands

| Command | What it does |
|---------|-------------|
| `/init` | Saves Groq + Supermemory API keys, detects GPU VRAM. Run once. |
| `/models-download` | Downloads CLIP, SmolVLM, BLIP-2, Whisper, and CrossEncoder to local cache. Run once. |
| `/index <path> smolvlm\|blip2` | Indexes a video extracts meaningful frames, captions them, transcribes audio, stores everything in ChromaDB. |
| `/init-llama` | Downloads Gemma 3 4B weights + llama-server binary. Required before `/index-llama`. Needs 4 GB+ VRAM. |
| `/index-llama <path>` | Indexes using local Gemma 3 via llama-server best caption quality, reads text off slides directly. |
| `/index-ollama <path> <model>` | Indexes using any Ollama vision model, just make sure you have the model downloaded |
| `/help` | Shows commands and VRAM guide inside the session. |
| `/clear` | Clears the terminal and redraws the banner. |
| `/exit` | Ends the session. |

Type your question directly at the prompt (no slash) to query the indexed video.
Answers are grounded in the audio transcript and frame captions from your specific video.

---

## Requirements

- Python 3.12
- [Groq API key](https://console.groq.com) — free tier available
- [Supermemory API key](https://supermemory.ai) — for conversation memory
- GPU with 4 GB+ VRAM recommended; CPU only mode works but is slower

## Project Structure

```
gurrt/
└── src/gurrt/
    ├── api/
    │   └── server.py          # API server (planned rewrite)
    ├── cli/
    │   ├── main.py            # CLI entry point and REPL
    │   └── ui.py              # All color constants, panels, Rich components
    ├── config/
    │   └── config.py          # API keys, model paths, session state
    ├── core/
    │   ├── asr.py             # Audio extraction + Whisper transcription
    │   ├── embedding.py       # Captioning + CLIP embedding per backend
    │   ├── llm.py             # Groq LLM chain + Supermemory conversation
    │   ├── models.py          # Model loading, caching, and release
    │   ├── pipeline.py        # End-to-end index and query orchestration
    │   ├── prompts.py         # LLM prompt templates
    │   ├── search.py          # Dual retrieval + CrossEncoder reranking
    │   └── vectordb.py        # ChromaDB interface
    └── utils/
        └── utils.py           # Temporal persistence filter, audio pipeline
```

#### Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Commit your changes**: `git commit -m "feat: describe your change"`
4. **Push to your fork**: `git push origin feature/your-feature-name`
5. **Open a Pull Request** against the `main` branch

Please ensure your code follows the existing project conventions and all existing functionality continues to work.

## License

This project is open-source and available under the [MIT License](LICENSE).

---