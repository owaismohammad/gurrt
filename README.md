[![PyPI version](https://img.shields.io/pypi/v/gurrt)](https://pypi.org/project/gurrt/) [![Python Versions](https://img.shields.io/pypi/pyversions/gurrt)](https://pypi.org/project/gurrt/) 
[![License](https://img.shields.io/pypi/l/gurrt)](https://pypi.org/project/gurrt/) [![Downloads](https://pepy.tech/badge/gurrt/)](https://pepy.tech/project/gurrt) [![Twitter Follow](https://img.shields.io/twitter/follow/muffBozo.svg?style=social)](https://twitter.com/muffBozo)
<p align="center">
  <img src="https://raw.githubusercontent.com/owaismohammad/gurrt/main/gurrt.png" width="450">
</p>

**gUrrT** (derived from the **Surveilens** research paper) is an optimized framework designed to bypass the heavy computational requirements of Large Video Language Models (LVLMs). While standard LVLMs often require high-end enterprise GPUs, gUrrT is engineered to deliver high-accuracy video understanding on consumer grade hardware (e.g., 4GB VRAM) by decomposing video into its core sensory components.

**The Philosophy: Pragmatic Decomposition**
With gUrrT, the goal isn't to reinvent the wheel or solve the complex "temporal dimension" problem that plagues modern AI. Instead, the project explores a critical question: Can we achieve "Video Understanding" simply by treating a video as a searchable collection of moments?

By bypassing the temporal modeling used in expensive LVLMs, gUrrT enables you to "talk to a video" by transforming it into a structured, queryable index. It gets the job done without the hefty compute tax.

The "Temporal Dimension" of video is computationally expensive to process directly. gUrrT shifts the paradigm from **Video Modeling** to **Contextual Retrieval**:

* **Vision Models (The Eyes):** Describe discrete scenes and frames.
* **Transcription Models (The Ears):** Process audio via Faster-Whisper.
* **Advanced Sampling:** Intelligently reduces the frame-load to only what is relevant.
* **RAG (The Brain):** Compiles these sensory inputs into a vector-based context for a Large Language Model (LLM).

---

### **The Technical Pipeline**

1. **Dual-Stage Frame Sampling:** * **Scene Detection:** The primary method, segmenting video into distinct events. For each scene, the pipeline captures the **start, middle, and end frames**.
* **Uniform Sampling:** Acts as a robust fallback if no distinct scene transitions are detected.
* *Note: SSIM (Structural Similarity Index) was tested but discarded to prioritize processing speed.*


2. **Multimodal Embedding:** * Visuals are embedded using **CLIP**, and captions are generated via **BLIP** (though experimentation shows BLIP’s limitations in context density).
* Audio is processed via **Faster-Whisper** and stored in a separate vector collection.


3. **Inference & LLM Integration:**
* The system supports local execution via **Ollama** (Gemma 3 performs exceptionally well) and cloud-based inference via **Groq** (utilizing Llama 3-70B for high-reasoning tasks).


4. **Supermemory:** * To prevent context "noise," the system utilizes a **Supermemory** feature that maintains a clean, video-specific context. It refreshes upon new video uploads to ensure response quality remains high and relevant to the current file.

---

### **Key Insights & Experimental Inferences**

* **The "Captioning Bottleneck":** The quality of the LLM’s response is directly proportional to the quality of the image-to-text descriptions. Upgrading from BLIP to more descriptive captioning models remains a primary goal for improving context.
* **Model Scaling:** Moving from **Llama 3.1-8B** to **Llama 3-70B** resulted in a phenomenal leap in performance. While the 8B model struggled with simple queries when fed BLIP data, the 70B model (and Gemma 3) demonstrated the "reasoning' necessary to synthesize poor context into accurate answers.
* **The Summary Challenge:** While RAG excels at specific "needle-in-a-haystack" queries, generating holistic video summaries remains a challenge for vanilla RAG architectures.

---

### **Future Roadmap**

I am looking into transitioning from **Vanilla RAG** to a **Graph-based RAG** or a **Hierarchical RAG** architecture. This would allow the system to understand the relationship between scenes over time, rather than treating them as isolated data points.

## 🌿 Quick Start Guide for pypi package

### 1. Installation

Set up **gurrt** using `uv`. Note: This project requires **Python 3.12**.

```bash
# 1. Install uv and set Python version
pip install uv
uv venv
uv python pin 3.12

# 2. Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install gurrt (Standard/CPU)
uv pip install gurrt

# 4. OR Install with GPU Support
uv pip install gurrt[cuda] --extra-index-url https://download.pytorch.org/whl/cu121

```

---

### 2. Commands

| Command | Description |
| --- | --- |
| `gurrt init` | Configure API keys (Groq, Supermemory, Ollama). |
| `gurrt models-download` | Download and cache AI models locally. |
| `gurrt index <path>` | Extract frames and audio for search. |
| `gurrt index-ollama <path> <model>` | Index using a specific Ollama model. |
| `gurrt ask "<query>"` | Query your indexed video content. |

The tool automatically optimizes performance by disabling unnecessary logging and tokenizer parallelism to ensure a clean CLI experience yet some logs do appear of Moviepy will resolve it in future iterations.

---


### Architecture Overview
```bash
Video
  │
  ├── Smart Frame Extraction
  │     └── Captioning + Embeddings
  │
  ├── Audio Extraction
  │     └── Speech-to-Text + Embeddings
  │
  ├── Vector Memory Store
  │
  ├── Supermemory (Persistent Conversation Layer)
  │
  └── LLM Reasoning Engine
```

### Project Setup (using uv)

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync

# Activate environment
.venv\Scripts\activate
```

### File Structure

```bash
gurrt/
├── src/
│   |
│   │
│   └── videorag/                      # Core Video-RAG application package
│       │
│       ├── api/
│       │   └── server.py              # API server (exposes endpoints for querying, ingestion, etc.)
│       │
│       ├── cli/
│       │   └── main.py                # CLI entry point (init, ingest, query commands)
│       │
│       ├── config/
│       │   └── config.py              # Configuration management (API keys, paths, environment setup)
│       │
│       ├── core/                      # Core intelligence pipeline
│       │   ├── __init__.py
│       │   ├── asr.py                 # Audio extraction + speech-to-text processing
│       │   ├── embedding.py           # Embedding generation for captions & transcripts
│       │   ├── llm.py                 # LLM interaction and reasoning logic
│       │   ├── models.py              # Model loading and management utilities
│       │   ├── pipeline.py            # End-to-end ingestion + query pipeline orchestration
│       │   ├── prompts.py             # Prompt templates and structured context injection
│       │   ├── search.py              # Retrieval logic (semantic search over stored embeddings)
│       │   └── vectordb.py            # Vector database interface and storage abstraction
│       │
│       └── utils/
│           └── utils.py            # Shared utility functions and helpers
│
└── README.md                         # Project documentation
```






















