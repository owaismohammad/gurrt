# gurrt
### An intelligent video understanding system designed as an open-source alternative to monolithic Large Video Language Models
I built gurrt out of frustration.

Working with Large Video Language Models locally is:

- Expensive to set up  
- GPU intensive  
- Slow to experiment with  
- Difficult to run on consumer hardware  
- Often closed or partially restricted  

Most state-of-the-art video models require massive compute clusters and large-scale infrastructure.  
They are impressive — but they are not accessible.

If meaningful video intelligence requires:

- Multiple high-end GPUs  
- Hours of inference time  
- Proprietary model access  

Then it stops feeling truly open.

---

### A Different Philosophy

gurrt is not an attempt to compete with systems like YouTube’s internal models or large-scale industrial LVLMs trained on massive GPU clusters.

It is an attempt to rethink the approach.

Instead of asking:

> "How can we train a bigger model?"

gurrt asks:

> "How can we design a smarter system?"

---

gurrt relies on:

- Modular components
- Retrieval-augmented reasoning  
- Semantic memory  
- Smart sampling  
- Persistent conversational context  


## Architecture Overview
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

## Project Setup (using uv)

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync

# Activate environment
.venv\Scripts\activate
```

## File Structure

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

