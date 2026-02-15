# gurrt: An intelligent video understanding system 

## An open-source alternative to monolithic Large Video Language Models built out of frustration.


One cannot work with Large Video Language Models :

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

gurrt is not an attempt to compete with systems like YouTube’s internal models or other large-scale industrial LVLMs trained on massive GPU clusters.
It is an attempt to rethink the approach.
Instead of asking how to build a larger end-to-end video transformer, it explores a different path:

- Smarter frame sampling techniques  
- Stronger and more modular vision models  
- Better structured embedding strategies  
- More efficient and grounded RAG pipelines  
- Persistent memory-driven reasoning  

The idea is how can i just get the job done with minimal efforts yielding high end results

It represents a belief that meaningful video understanding can emerge from:

- Thoughtful engineering  
- Smart sampling  
- Strong modular components  
- Memory-augmented retrieval  

Not just from massive GPU clusters and billion-parameter models.


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





