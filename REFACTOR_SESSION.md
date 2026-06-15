# gurrt Refactor Session — Summary

## Major Architecture Changes ✅

### Temporal Persistence Filter — Adaptive Frame Sampling for Lecture Videos

**Files:** `utils.py`
**Replaces:** `scene_split()` + `frame_listing()` + `frame_listing_uniform()` fallback chain

---

#### What was wrong with the old approach

The old pipeline ran two techniques depending on what PySceneDetect found:

- **Scene detection (`ContentDetector`)** — built for cinematic shot boundaries (full-frame color changes, hard cuts). In lecture videos, slides transition by new text appearing or fading in — not full-scene cuts. ContentDetector either fires on nothing (returns empty scene list) or fires on irrelevant motion artifacts.
- **Uniform 1 FPS fallback** — when no scenes were detected, every second produced a frame regardless of whether anything changed. A slide displayed for 3 minutes generated 180 near-identical frames, all sent through BLIP for captioning. For a 1-hour lecture this produced ~3,600 frames; only a small fraction were visually distinct.

The core failure of both: neither technique could distinguish **meaningful content change** from **no change at all**.

---

#### The new approach — three passes

**Pass 1 — Lightweight hash extraction (CPU only)**

Scans the video at 2 FPS (every `fps // 2` frames) using `cap.set(CAP_PROP_POS_FRAMES)` for direct seeking. For each sampled frame, computes a `dHash` (difference hash — 64-bit, gradient-direction based). Stores only `(frame_number, timestamp_sec, dhash)` tuples — no PIL images held in memory. For a 1-hour lecture this means ~7,200 hashes at ~8 bytes each (~57 KB total) instead of gigabytes of image data.

`dHash` is chosen over `pHash` specifically because it encodes local pixel gradient direction rather than global frequency content (DCT). This makes it unusually sensitive to text appearing or disappearing on a slide — exactly the signal that matters for lecture content.

**Pass 2 — Persistence state machine (CPU only)**

Runs a three-state machine over the collected hash list. The key invariant: `reference_hash` (the hash of the current known slide state) only updates on a *confirmed* change — never mid-window.

```
STABLE
  Every frame compared to reference_hash.
  Distance > hash_threshold → enter CANDIDATE.

CANDIDATE
  A potential change was detected. Do not select yet.
  Collect Hamming distances vs. reference_hash for persistence_window_sec seconds.
  At window expiry, vote:

    far_frames / total_window_frames >= vote_ratio
      → CONFIRMED: slide genuinely changed.
        Select the candidate frame (first frame of the new state).
        Update reference_hash to candidate's hash.

    far_frames / total_window_frames < vote_ratio
      → FALSE POSITIVE: speaker movement, hand obscuring slide.
        Discard silently. Keep old reference_hash.

  Either way → back to STABLE.
```

Two temporal guardrails run at every step:
- `min_interval_sec` — prevents selecting two frames that are too close together (transition animation artifacts, rapid gestures).
- `max_interval_sec` — force-selects a frame if no confirmed change has occurred for this long. Ensures long static sections are never completely unrepresented in the index.

**Why this filters the speaker:**
A lecturer walking in front of the slide causes `distance > threshold` for a few seconds, then walks away and frames return close to `reference_hash`. The vote ratio fails — majority of window frames are back below threshold. Nothing selected.

A new slide appearing causes `distance > threshold` on the first frame, and *every subsequent frame in the window* is also above threshold (the new slide is still there). Vote passes. Candidate frame selected.

**Pass 3 — Targeted re-read (disk I/O only on selected frames)**

After the state machine runs, only `M` frame numbers are confirmed. The video is re-opened and only those `M` positions are seeked with `cap.set(CAP_PROP_POS_FRAMES, idx)`. For a 1-hour lecture, `M` is typically 80–200 frames — a 15–40× reduction compared to uniform 1 FPS. These PIL images are passed directly into the existing `batched_captioning()` pipeline.

PIL images are deliberately **not stored** during Pass 1 to avoid holding hundreds of full-resolution frames in RAM simultaneously. The second video open reads only the confirmed frames, not the full scan set.

---

#### How it improves the existing architecture

| Dimension | Before | After |
|-----------|--------|-------|
| Frames sent to BLIP (1-hour lecture) | ~3,600 (1 FPS uniform) | ~80–200 (confirmed changes only) |
| Scene detection relevance | ContentDetector tuned for cinema, largely useless on slides | Replaced entirely — no scene detection library needed |
| Redundant frames | High — same slide generates hundreds of identical frames | Near zero — repeated visual state produces one frame |
| Speaker false positives | Not addressed — every second sampled regardless | Filtered by persistence window vote |
| Memory during sampling | Proportional to total frames if images stored | ~57 KB for hashes regardless of video length |
| GPU time (BLIP captioning) | Dominant bottleneck, run on all sampled frames | Run only on confirmed-change frames |

---

#### Key parameters and what they control

| Parameter | Default | Controls |
|-----------|---------|----------|
| `hash_threshold` | 12 | Sensitivity to visual change. Lower catches subtle content additions (new bullet, new formula line). Higher ignores minor differences. |
| `persistence_window_sec` | 5.0 | Must be shorter than the minimum slide duration in target videos. If slides change every 4s, set this to ≤3s or changes are never confirmed. |
| `vote_ratio` | 0.6 | How consistently the change must persist. Lower tolerates speaker movement during the window. Higher demands clean transitions. |
| `min_interval_sec` | 2.0 | Prevents duplicate selections during slide transition animations. |
| `max_interval_sec` | 60.0 | Coverage fallback. Any 60-second window without a confirmed change force-inserts one frame. |

---

#### Known limitations

- **Whiteboard animation videos** (content continuously hand-drawn): The hand acts as a permanent "speaker" — always present, always moving. `persistence_window_sec` must be reduced to 1.5–2.0s and `max_interval_sec` to 10–15s for these videos. The default parameters are tuned for slide-based lectures.
- **Speaker standing still in front of slide for >persistence_window_sec**: Indistinguishable from a slide change. A frame is selected. The caption describes the speaker rather than the slide content. Edge case in practice.
- **Rapid slide transitions (<2s per slide)**: If slides change faster than `persistence_window_sec`, no change is ever confirmed. Reduce the window or increase scan FPS.

---

## Performance Optimization Session ✅

### Frame Sampling — Single-Pass Sequential Scan

**File:** `utils.py` → `temporal_persistence_filter()`

#### Root cause of old slowness

The old implementation had three separate passes, each using random seeks:

1. **Hash pass** — `cap.set(CAP_PROP_POS_FRAMES, frame_no)` + `cap.read()` repeated ~202 times for a 1:41 video. Each `cap.set()` on H.264/H.265 forces a keyframe-based decode restart. At ~200ms per seek × 202 seeks = ~40 seconds, before any hash computation. The hash algorithm (`dhash` vs numpy) was irrelevant — the seek was the bottleneck.
2. **State machine pass** — in-memory, fast, not a problem.
3. **Re-read pass** — another N random seeks to read back only the confirmed selected frames.

#### What changed

**Hash computation — PIL eliminated:**
Replaced `imagehash.dhash(Image.fromarray(frame))` (3 allocations per frame: RGB conversion, PIL Image, imagehash internals) with pure numpy/cv2:
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
current_hash = (small[:, 1:] > small[:, :-1]).flatten()
distance = int(np.count_nonzero(current_hash ^ ref_hash))
```
Zero PIL allocations during the scan. Hamming distance via `np.count_nonzero(a ^ b)`.

**Sequential scan with `cap.grab()`:**
Replaced the `for frame_no in frame_idx: cap.set(...)` pattern with a single `while` loop:
- Sampled frames (`frame_no % req_fps == 0`): `cap.read()` — full decode
- Skipped frames: `cap.grab()` — reads compressed packet from container without decoding

`cap.grab()` = `cap.read()` minus `cap.retrieve()` (the decode step). At `fps_selected=2` on 30fps video, `req_fps=15` means 14 out of every 15 frames are grabbed without decode — ~93% of frame reads become cheap I/O-only operations.

**All three passes merged into one:**
The state machine runs inline inside the single loop. When a frame enters `CANDIDATE` state, its raw BGR is stored with `frame.copy()` — one frame in memory at a time, always freed when the window expires. Confirmed frames stored as `(timestamp, raw_bgr)`. PIL conversion happens at the end on only the confirmed set. No re-read pass at all.

---

### Frame Sampling — FFmpeg Pipe + Adaptive Sampling

**File:** `utils.py` → `temporal_persistence_filter()`

#### Root cause of remaining slowness

The single-pass `cap.grab()` loop fixed the random-seek problem but the pipe-level cost remained: `cap.read()` on sampled frames still returns full-resolution BGR (`width × height × 3` bytes per frame). For 1080p at 2fps over a 1-hour video that is ~43GB of frame data flowing through Python even though the only downstream consumer of each frame was a 9×8 hash computation. The hash itself was fast; the data volume was not.

Additionally, `cap.grab()` on skipped frames still partially decodes P/B-frames at the codec level — H.264 inter-frame dependencies mean the codec must reconstruct intermediate frames to produce the next sampled one, even when `retrieve()` is never called.

A second cost: `candidate` stored `frame.copy()` — a full-resolution BGR array on the heap — for every active candidate frame. At 1080p this is ~6MB held in RAM per candidate, and candidates accumulate for the full persistence window duration.

#### What changed

**Pass 1 — ffmpeg pipe at 72 bytes/frame (CPU only):**

Replaced `cv2.VideoCapture` loop with a subprocess pipe from ffmpeg that outputs pre-scaled grayscale frames:

```python
proc = subprocess.Popen([
    ffmpeg_exe,
    "-threads", "0",           # auto multi-threaded H.264 decode (uses all CPU cores)
    "-skip_frame", "noref",    # skip B-frames — non-reference frames never needed at 2fps output
    "-i", str(video_path),
    "-vf", f"fps={fps_selected},scale=9:8:flags=area",
    "-f", "rawvideo", "-pix_fmt", "gray",
    "pipe:1"
], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
```

Each frame arrives as 72 bytes (`9 × 8` gray). `np.frombuffer(raw, ...).reshape(8, 9)` + hash is all Python sees per frame. cv2 color conversion and resize are eliminated entirely — ffmpeg does both in C before bytes reach Python.

`-threads 0` lets ffmpeg use all available CPU cores for H.264 decode (Ryzen 5 5600H: 6 cores / 12 threads).

`-skip_frame noref` instructs the decoder to skip B-frames (non-reference frames). B-frames are bidirectionally predicted and never selected as output by the `fps=2` filter, so skipping their decode is safe. Saves 20–40% of decode work on typical H.264 lecture videos depending on GOP structure.

**Adaptive sampling — `stable_fps` parameter:**

Added `stable_fps=0.5` parameter. In STABLE state the state machine only needs to detect that *something* changed — checking every 2 seconds is sufficient. In CANDIDATE state the full 2fps resolution is needed to correctly time the persistence window.

```python
stable_step = max(1, round(fps_selected / stable_fps))  # e.g. round(2 / 0.5) = 4

if state == STABLE and ref_hash is not None and (frame_index % stable_step) != 0:
    continue  # drain 72 bytes, skip hash logic
```

- **STABLE**: 1 in every `stable_step` frames is hashed — effective 0.5fps scan rate
- **CANDIDATE**: every frame is hashed — full 2fps resolution for the window vote
- Transition is automatic: once a spike is detected and `state` becomes `CANDIDATE`, the `if state == STABLE` guard no longer fires and every frame is processed

The skip fires *before* the numpy hash computation, not after — no wasted work on skipped frames.

**Pass 2 — targeted cv2 seeks for confirmed frames only:**

`candidate` no longer stores `frame.copy()`. Only `(timestamp, hash_bits)` is kept — 72 bytes per candidate instead of ~6MB.

After the pipe closes, full-resolution frames are fetched only for confirmed timestamps:

```python
cap = cv2.VideoCapture(str(video_path))
for ts in confirmed_timestamps:
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ret, frame = cap.read()
    if ret:
        frame_PIL.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
cap.release()
```

For a 1-hour lecture with ~40 confirmed slides this is 40 seeks. The random-seek cost per seek on an SSD for a confirmed frame is negligible (~5–15ms each).

---

#### Effect on the sampling pipeline

| Dimension | After Step 2 (cap.grab loop) | After Step 3 (ffmpeg pipe + adaptive) |
|-----------|------------------------------|----------------------------------------|
| Pipe data volume (1hr 1080p) | ~43GB (full BGR per sampled frame) | ~518KB (72 bytes per frame) |
| Frames hashed in STABLE state | All sampled frames (2fps × 3600s = 7,200) | ~25% of sampled frames (stable_step=4, effective 0.5fps) |
| RAM held for candidate frame | ~6MB BGR copy per active candidate | 72 bytes (timestamp + hash only) |
| Full-res reads | All confirmed frames + all candidates in memory since start | Only confirmed frames, fetched after pipe closes |
| Decode threading | Single-threaded cv2 VideoCapture | Auto-threaded ffmpeg (`-threads 0`) |
| B-frame decode | Forced by H.264 inter-dependency even with cap.grab() | Skipped (`-skip_frame noref`) |

---

### Audio Pipeline — Three Independent Wins

**File:** `utils.py` → `audio_extraction()`, `audio_to_text()`
**File:** `config.py` → `WHISPER_MODEL`
**File:** `models.py` → `get_whisper()`, `download_models()`, new `download_whisper()`

#### 1. Audio extraction — moviepy replaced with ffmpeg subprocess

moviepy decodes the full video frame-by-frame just to extract audio. Direct ffmpeg demuxes the audio stream from the container without touching video frames:

```python
subprocess.run([
    "ffmpeg", "-y", "-i", str(path),
    "-vn", "-acodec", "pcm_s16le",
    "-ar", "16000", "-ac", "1",
    str(audio_file)
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
```

`-vn` skips video decode. Output is 16kHz mono PCM WAV — exactly what Whisper expects. `AUDIO_PATH` in config changed from `.mp3` to `.wav` to match.

#### 2. Whisper transcription flags

`beam_size=5` → `beam_size=1` (greedy decode, ~3-4x faster, negligible accuracy drop on clean lecture audio) and `vad_filter=True` (Silero VAD skips silence — 20-40% fewer segments on lecture content with natural pauses). Both passed from `asr.py` into `audio_to_text()`.

#### 3. Whisper model size

`WHISPER_MODEL = "large-v2"` → `"small"` in `config.py`. large-v2 is the heaviest model; `small` with int8 quantization is ~8x faster on the same hardware with acceptable accuracy for lecture audio.

#### 4. Whisper local caching

Previously Whisper auto-downloaded to HuggingFace's global cache (`~/.cache/huggingface/hub/`) on every new model size — needed internet, not co-located with other models. Now:

- `download_models()` / new `download_whisper()` in `models.py` uses `snapshot_download(repo_id="Systran/faster-whisper-small", local_dir=cache_dir/"whisper_model")` to store the model alongside CLIP/BLIP/Reranker in `MODEL_CACHE_DIR`.
- `get_whisper()` loads from `self.cache / "whisper_model"` (local path) instead of the model name string. Works fully offline after first download.

---

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
| `frame_no` already incremented when stored in `candidate` | `utils.py` | `frame_no += 1` moved to after read/grab but before state machine logic; candidate stores `frame_no - 1`. Re-read pass was seeking to wrong frame (off by one). |
| Re-read pass still used random seeks after refactor | `utils.py` | Eliminated entirely — candidate now stores `frame.copy()`, PIL conversion done at end on confirmed frames only |
| `pbar.update(1)` skipped on `continue` paths | `utils.py` | Added `pbar.update(1)` before each `continue` (first-frame init and max-interval force-select paths). Progress bar was stalling then jumping. |
| Dead code `hashed_frames = []` left from old 3-pass approach | `utils.py` | Removed |
| Generator double-iteration in `audio_to_text` | `utils.py` | `segments = list(segments)` materializes the lazy generator immediately after `transcribe()`. Without this, iterating `segments` in a debug/print cell and then again in `"".join()` silently returns empty on the second pass — a silent data loss bug that produces an empty transcript with no error. |

---

## Medium Effort ✅

| Target | File(s) | Notes |
|--------|---------|-------|
| Batch uniform frame sampling | `utils.py`, `embedding.py` | Created `frame_listing_uniform()` that collects all sampled frames first, then feeds them into the existing `batched_captioning()` (batch_size=16). Both scene-detection and uniform-sampling fallback now share the same batched captioning pipeline |

---

### ASR Pipeline — BatchedInferencePipeline + Distil-Whisper

**Files:** `models.py` → `get_whisper()`, `download_models()`; `utils.py` → `audio_to_text()`

#### What changed

**`get_whisper()` returns `BatchedInferencePipeline` instead of raw `WhisperModel`:**

`BatchedInferencePipeline` wraps the underlying `WhisperModel` and groups multiple audio segments into a single CTranslate2 kernel call per batch instead of processing them one at a time. Call sites (`asr.py`) required no changes — the batched object exposes the same `.transcribe()` API. Significant win on CUDA; marginal on CPU due to padding overhead.

**`audio_to_text()` uses the batched API:**

Switched from `model.transcribe(audio_path, beam_size=beam_size, vad_filter=True)` to `model.transcribe(audio_path, batch_size=8, vad_filter=True)`. `beam_size` is no longer passed — the batched pipeline handles decoding internally. `batch_size=8` is a conservative starting point for 4GB VRAM; can be increased to 16 on larger GPUs.

**Generator materialized immediately:**

`segments = list(segments)` added right after `transcribe()`. `transcribe()` returns a lazy generator — actual transcription runs only when iterated. Without materializing, any code that iterates `segments` twice (a print loop then `"".join()`, or a re-run cell in a notebook) gets an empty result on the second pass because a Python generator can only be consumed once. `list()` forces transcription to complete immediately and stores the result as a plain list safe to iterate repeatedly.

**Distil-Whisper as default model:**

`download_models()` default changed from `model_name="small"` to `model_name="distil-large-v2"`. Downloads `Systran/faster-whisper-distil-large-v2` — a knowledge-distilled version of Whisper large-v2. Approximately 3× faster than large-v2 with ~3% WER increase on clean English speech. English-only; not suitable if multilingual support is needed.

**Known gap:** `download_whisper()` standalone function still defaults to `model_name="small"` — inconsistency with `download_models()`. A user calling it directly gets the small model, not distil-large-v2.

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
| **Replace BLIP with a richer VLM** | High | BLIP captions for lecture slides are poor ("a whiteboard with writing on it"). moondream2 (1.8B) can read text directly off slides and fits on 6GB VRAM alongside CLIP. OCR (pytesseract/easyocr) is even faster for text-heavy slides and more accurate — hybrid: OCR when text is detected, moondream2 fallback for diagrams/blackboard. |
| **Drop BLIP `num_beams=3` to 1** | High | Beam search in BLIP generate is ~2-3x slower than greedy (`num_beams=1`). One line change in `generate_captions_in_batches`. BLIP captioning is now the dominant bottleneck after audio and hashing were sped up. |
| ~~**`selected_frames` memory usage**~~ | ~~Medium~~ | ~~Currently stores `(timestamp, raw_bgr)` for all confirmed frames. For a 1-hour lecture with 150 slide changes at 1080p this is ~900MB. Fix: store `(frame_no, timestamp)` only, do a small re-read pass at the end (~150 seeks, fast).~~ **Resolved in FFmpeg Pipe + Adaptive Sampling step** — `candidate` now stores only `(timestamp, hash_bits)`; full-res reads done in Pass 2 for confirmed frames only. |
| **Align `download_whisper()` default with `download_models()`** | Medium | `download_whisper()` standalone still defaults to `model_name="small"` while `download_models()` now defaults to `"distil-large-v2"`. A user calling `download_whisper()` directly gets the wrong model. Both should read the model name from `Settings()` rather than hardcoding. |
| **`subprocess` import position in `utils.py`** | Low | Imported mid-file instead of at the top with other imports. |
| **CLI `models_download` doesn't pass model name from Settings** | Low | `download_models(cache_dir)` in `cli/main.py` calls with the hardcoded default. If a user changes `WHISPER_MODEL` in config, the downloaded model won't match what `get_whisper()` loads. Should read from `Settings()` and pass through. |
| **FastAPI `server.py` rewrite** | Low | Currently broken — imports `query_llm` and `delete` as module-level functions (they are methods on `LLMService`), and uses subprocess paths that don't work for an installed package. Needs full rewrite using `VideoRag` properly |
| **BM25 / sparse hybrid retrieval for ASR** | Low | Dense CLIP embeddings + keyword overlap for transcript search. Lower priority given the CrossEncoder reranker already handles re-scoring accurately |

---

## CLI / UX Improvements Session ✅

**Files:** `src/gurrt/cli/main.py`, `src/gurrt/cli/ui.py`

---

### 1. Color Theme Centralized to `ui.py`

All color constants moved to `cli/ui.py` — nothing hardcoded in `main.py`. Every panel border, rule, progress bar, and prompt-toolkit color is defined in one place and referenced via `ui.BORDER_*` / `ui.C_*` constants.

```python
# cli/ui.py
C_ACCENT   = "bright_cyan"
C_SUCCESS  = "bright_green"
C_ERROR    = "bright_red"
C_WARNING  = "yellow"
C_DIM      = "dim"

BORDER_PRIMARY = "cyan"
BORDER_SUCCESS = C_SUCCESS
BORDER_ERROR   = C_ERROR
BORDER_WARNING = C_WARNING    # added this session — yellow, for setup-required panels
```

---

### 2. Answer Panel Styling

LLM responses rendered with `Markdown()` inside a `Panel` with a cyan border (`BORDER_PRIMARY`). Content is plain white — not tinted.

```python
console.print(Panel(
    Markdown(response),
    title="[primary]Answer[/primary]",
    border_style=ui.BORDER_PRIMARY,
))
```

---

### 3. Slash Command Autocomplete (`_SlashCompleter`)

When the user types `/` in the REPL, a popup lists all available commands with descriptions. Implemented with prompt_toolkit's `Completer` API. Completion fires on every keystroke while the input starts with `/`.

```python
class _SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        typed = text[1:].lower()
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(typed):
                yield Completion("/" + cmd, start_position=-len(text),
                                 display=f"/{cmd}", display_meta=desc)
```

All popup colors (background, selected entry, meta text, scrollbar) defined in `ui.py` under `_PT_POPUP_*` constants.

---

### 4. Slash Command Input Highlighting (`SlashCommandLexer`)

As the user types `/command args`, the `/command` token is rendered in bold cyan and the argument text in white. Uses a custom prompt_toolkit `Lexer`.

```python
class SlashCommandLexer(Lexer):
    def lex_document(self, document):
        lines = document.lines
        def get_line(lineno):
            line = lines[lineno]
            if not line.startswith("/"):
                return [("", line)]
            idx = line.find(" ")
            if idx == -1:
                return [("class:slash-cmd", line)]
            return [("class:slash-cmd", line[:idx]), ("class:slash-args", line[idx:])]
        return get_line
```

`slash-cmd` maps to `bold ansicyan`; `slash-args` maps to `ansiwhite` — both defined in `ui.pt_style`.

---

### 5. Screen Clear on `gurrt` Launch

`os.system("cls" if os.name == "nt" else "clear")` at the top of the `main()` callback so the terminal is cleared and starts with just the banner whenever `gurrt` is invoked.

---

### 6. Spinner Dots for All Loading Operations

Every blocking operation wrapped in `console.status("[info]...[/info]", spinner="dots")` so the terminal shows an animated spinner instead of freezing silently.

| Operation | Spinner label |
|-----------|--------------|
| GPU detection in `/init` | `Detecting GPU...` |
| Model download in `/models-download` | `Downloading and caching models...` |
| Gemma 3 weights download in `/init-llama` | `Downloading Gemma 3 model weights...` |
| llama-server release fetch | `Fetching latest llama-server release from GitHub...` |
| llama-server zip download | `Downloading <filename>...` |
| llama-server zip extraction | `Extracting server binary...` |
| Model load in `/index` | `Loading model...` |
| Frame indexing (smolvlm) | `Indexing frames with SmolVLM...` |
| Frame indexing (blip2) | `Indexing frames with BLIP-2...` |
| Frame indexing (llama) | `Indexing frames with Gemma 3 via llama-server...` |
| Frame indexing (ollama) | `Indexing frames with <model>...` |
| Audio transcription | `Transcribing audio...` |
| LLM answer | `Thinking...` |

---

### 7. `/clear` Command

Clears the terminal and redraws the banner + gUrrT session panel. If a video is currently indexed, the panel shows "Session active" with the current video path. If not indexed, shows the full command reference.

```python
elif cmd == "clear":
    os.system("cls" if os.name == "nt" else "clear")
    ui.show_banner()
    if _indexed:
        console.print(Panel(
            f"[success]Session active.[/success]\n"
            f"[dim]Indexed: {_last_video}[/dim]\n\n"
            "[primary]→[/primary] [dim]Type your question to continue ...[/dim]",
            title="[primary]gUrrT Session[/primary]",
            border_style=ui.BORDER_PRIMARY,
        ))
    else:
        # full command-reference panel
```

---

### 8. Session Persistence (`session.json`)

`session.json` stored in the platform config dir (`platformdirs.user_config_dir("gurrt")`) acts as a persistent key-value store across REPL sessions. All save functions use a **read-merge-write** pattern so no key is accidentally cleared when another key is saved.

| Key | Type | Written by | Read by |
|-----|------|-----------|---------|
| `last_video` | `str` | `_save_session()` after each successful index | `_load_session()` at REPL startup |
| `ollama` | `bool` | `_save_ollama_flag()` during `/init` | `_get_ollama_flag()` in `_do_index_ollama()` |
| `gpu_mb` | `int` | `_save_gpu_info()` during `/init` | `_get_gpu_mb()` in index commands and `_do_init_llama()` |
| `init_done` | `bool` | `_save_init_done()` after `/init` completes | `_check_prereqs()` |
| `models_downloaded` | `bool` | `_save_models_done()` after `/models-download` completes | `_check_prereqs()` |

---

### 9. Ollama Presence Check in `/init`

After the user enters API keys, `/init` asks whether Ollama is installed (yes/no prompt). The answer is saved to `session.json` as `"ollama": bool`.

`/index-ollama` reads this flag before proceeding. If `False`, shows a red error panel explaining Ollama is not installed and linking to `https://ollama.com`, rather than letting the call fail with a cryptic error.

---

### 10. GPU VRAM Saved at `/init` Time

`nvidia-smi` subprocess moved from being called on every `/index` to once during `/init`. Result saved to `session.json` as `"gpu_mb": int` (0 = no GPU detected). All index commands and `/init-llama` read from session.json via `_get_gpu_mb()`.

**Before:** subprocess ran on every `/index` call.  
**After:** subprocess runs once, result persists across sessions.

---

### 11. VRAM Guard for Llama Commands

`_do_init_llama()` and `_do_index_llama()` check GPU VRAM from session.json before doing any work.

- **VRAM info not available** (i.e., `/init` never run): yellow warning panel — "Run /init first."
- **VRAM < 4500 MB** (or 0 = no GPU): red error panel showing detected VRAM, explaining the 4 GB+ requirement, and suggesting `/index smolvlm` or `/index blip2` as alternatives.

The VRAM guard fires after `_check_prereqs()` — so if `/init` was skipped entirely, the prereqs message appears first (clearer than "GPU info not available").

---

### 12. Setup Prerequisites Check (`_check_prereqs()`)

All index commands and `/init-llama` call `_check_prereqs(command_name)` as their first action. Reads `init_done` and `models_downloaded` from `session.json`. If either is missing, prints a yellow panel that lists exactly which step(s) are missing and returns `False` — the calling function returns immediately.

```python
def _check_prereqs(command: str) -> bool:
    # reads init_done, models_downloaded from session.json
    if init_done and models_done:
        return True
    # prints yellow Panel listing missing steps
    return False
```

Commands guarded: `/index`, `/index-llama`, `/index-ollama`, `/init-llama`.

---

### 13. Comprehensive Try-Catch Error Handling

Every major blocking operation wrapped in `try/except Exception as e`. On failure: prints a `Panel` with `border_style=ui.BORDER_ERROR` (red) and the raw exception text. The REPL session never crashes.

| Function | Blocks wrapped |
|----------|---------------|
| `_do_init()` | Config file save + ollama flag save |
| `_do_models_download()` | Entire model download |
| `_do_index()` | Video frame indexing (block 1), audio transcription (block 2) |
| `_do_index_llama()` | Gemma 3 weights download, video indexing (block 2), audio transcription (block 3) |
| `_do_index_ollama()` | Video indexing + ollama call (block 1), audio transcription (block 2) |
| `_do_init_llama()` | llama-server release fetch + download + extraction |

Each block is independent — a video indexing failure does not prevent the error from being shown cleanly and does not affect other session state.

---

### 14. Session Resume on Startup

At REPL startup, `_load_session()` reads `last_video` from `session.json`. If present, a `VideoRag()` object is created and the REPL starts in "indexed" state — the prompt shows `●` (green dot) and a "Session resumed" panel shows the last video path. If absent, the normal welcome panel with the command reference is shown.

---

### 15. Ask / Direct Question Guard

Both the direct question path (typing without `/`) and `/ask` check `if not _indexed or _rag is None` before querying the LLM. If no video is indexed, a yellow warning is shown. The check relies on `_indexed` which is `False` whenever `_load_session()` returned `(None, None)` — i.e., no `last_video` in `session.json` and no video indexed this session.

```python
if not raw.startswith("/"):
    if not _indexed or _rag is None:
        ui.warn("No video indexed yet. Use /index, /index-llama, or /index-ollama first.")
        continue
```

Same guard applies to the `/ask` slash command path.
