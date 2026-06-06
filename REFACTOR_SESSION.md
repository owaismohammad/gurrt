# gurrt Refactor Session — Summary

## Quick Wins ✅

| Target | File(s) | Notes |
|--------|---------|-------|
| Cache CrossEncoder — stop loading from disk per query | `pipeline.py`, `search.py`, `utils.py` | Loaded once in `VideoRag.__init__`, passed into `SearchService` and `rerank`/`rerank_docs` |
| Frame jumping in uniform sampling | `utils.py` | Replaced read-every-frame while-loop with `range(0, total_frames, int(fps))` + `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)` |
| Chunk size increase | `utils.py` | 100 chars → 300 chars / overlap 40. Capped at 300 (not 512) to stay within CLIP's 77-token hard limit and avoid silent truncation |
| Supermemory query pollution fixed | `llm.py` | Removed per-query `.add()` calls for frame/audio captions. Replaced with conversation memory — query+answer saved under `Previous_Chat` tag and retrieved as context for follow-up questions |
| BLIP generation params re-enabled | `utils.py` | `num_beams=3`, `min_length=15` in `generate_captions_in_batches` and `uniform_frame_sampling` |
| `truncation=True` in CLIP processor | `asr.py` | Explicit truncation at 77 tokens; prevents silent overflow errors |

---

## Bug Fixes ✅

| Bug | File | Fix |
|-----|------|-----|
| `tqdm` progress bar wrong total in Ollama uniform sampling | `utils.py` | `total=total_frames` → `total=len(frames_idx)` |
| BLIP params commented out in `uniform_frame_sampling` | `utils.py` | `num_beams=3`, `min_length=15` re-enabled |
| `delete()` clearing wrong Supermemory tags | `llm.py` | Replaced `Frame_Captions` + `Audio_Captions` deletion with `Previous_Chat` deletion |
| `source_path` stored as `Path` object in metadata | `utils.py`, `embedding.py` | Wrapped with `str()` across all three sampling functions |
| `video_path` stored as `Path` object in ASR metadata | `asr.py` | `str(video_path)` |

---

## Medium Effort ✅

| Target | File(s) | Notes |
|--------|---------|-------|
| Batch uniform frame sampling | `utils.py`, `embedding.py` | Created `frame_listing_uniform()` that collects all sampled frames first, then feeds them into the existing `batched_captioning()` (batch_size=16). Both scene-detection and uniform-sampling fallback now share the same batched captioning pipeline |

---

## Discarded Targets — with Reasons

| Target | Reason Discarded |
|--------|-----------------|
| **Replace CLIP text encoder with sentence-transformers for ASR** | We use a single CLIP query embedding to search both the frame collection and the ASR collection at once. If we switched ASR to sentence-transformers, we'd need two separate query embeddings at search time — one for frames, one for audio — which adds complexity. More importantly, the chunk size fix (300 chars) already solves the core quality problem by ensuring chunks fit within CLIP's limit, and the CrossEncoder reranker handles any remaining imprecision. The gain wasn't worth the added complexity. |
| **Parallelize `index_video` + `index_audio` with `ThreadPoolExecutor`** | Attempted but ran into a library conflict. faster-whisper (audio) and PyTorch (video frames) both try to access the GPU during startup, and when run in parallel threads they crash into each other. Even forcing audio to run on CPU didn't fix it — faster-whisper still checks for the GPU at the library level regardless. The workaround would have been too messy, so we reverted to running them sequentially. |
| **Batch Ollama uniform frame sampling** | Not applicable — Ollama takes one image at a time and gives back one caption per call. Sending multiple images together gives one combined caption for all of them, not individual captions per frame. |

---

## Remaining / Not Yet Implemented

| Target | Priority | Notes |
|--------|----------|-------|
| **HyDE (Hypothetical Document Embeddings)** | High | At query time, ask the LLM to generate a hypothetical frame caption for the query, embed it with CLIP, use that embedding for retrieval instead of the raw query. Bridges the phrasing gap between user questions and BLIP captions |
| **Replace BLIP with a richer VLM** | Medium | BLIP-base/large produces short generic captions. LLaVA via Ollama (`index_video_ollama`) is already partially available but only as an alternative path, not the default |
| **FastAPI `server.py` rewrite** | Low | Currently broken — imports `query_llm` and `delete` as module-level functions (they are methods on `LLMService`), and uses subprocess paths that don't work for an installed package. Needs full rewrite using `VideoRag` properly |
| **BM25 / sparse hybrid retrieval for ASR** | Low | Dense CLIP embeddings + keyword overlap for transcript search. Lower priority given the CrossEncoder reranker already handles re-scoring accurately |
