from pathlib import Path
import numpy as np
from typing import Dict, Any
from ollama import chat
import torch
import cv2
from io import BytesIO
from PIL import Image
import subprocess
import imageio_ffmpeg
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gurrt.config.config import Settings
from gurrt.core.prompts import VLM_PROMPT
from gurrt.cli import ui
def audio_extraction(path: Path, settings: Settings):
    audio_file = settings.AUDIO_PATH
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_exe, "-y", "-i", str(path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(audio_file)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_file

# def audio_to_text(audio_path, model, beam_size : int = 5) -> str:
#     segments, info = model.transcribe(audio_path, batch_size=8, vad_filter=True)
#     segments = list(segments)
#     text = "".join(segment.text for segment in segments)
#     return text
# def chunk_text(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 300,
#         chunk_overlap = 40
#     )
#     chunked_text = text_splitter.split_text(text=text)
#     return chunked_text
def audio_to_segments(audio_path, model, beam_size: int = 1):
    """Transcribe and keep per-segment timing instead of collapsing to one string."""
    segments, info = model.transcribe(audio_path, batch_size=8, vad_filter=True)
    return [(seg.start, seg.end, seg.text) for seg in segments]


def chunk_segments(segments, max_chars: int = 1000, overlap_segments: int = 1):
    """Group whisper segments into chunks that carry a start/end timestamp.

    Overlap is measured in segments rather than characters so chunks never
    split mid-sentence.
    """
    chunks, buf = [], []
    for seg in segments:
        buf.append(seg)
        if sum(len(s[2]) for s in buf) >= max_chars:
            chunks.append({
                "text": "".join(s[2] for s in buf).strip(),
                "start_sec": buf[0][0],
                "end_sec": buf[-1][1],
            })
            buf = buf[-overlap_segments:] if overlap_segments else []
    if buf:
        chunks.append({
            "text": "".join(s[2] for s in buf).strip(),
            "start_sec": buf[0][0],
            "end_sec": buf[-1][1],
        })
    return chunks
def embed_texts(texts: list, embedder, batch_size: int = 32) -> list:
    """Embed text for the vector DB.

    Frames and transcript chunks live in separate collections but are searched
    with one query vector, so both must be embedded by this same model.
    """
    if not texts:
        return []
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def generate_captions_in_batches(batch_of_frames,
                                 smol_model,
                                 smol_processor,
                                 device):
    messages_batch = [
        [{"role": "user","content": [{"type": "image"},{"type": "text", 
        "text": VLM_PROMPT}]}]
    for _ in batch_of_frames]
    prompts = [
        smol_processor.apply_chat_template(m, add_generation_prompt=True)
        for m in messages_batch
    ]
    smol_inputs = smol_processor(
        text=prompts,
        images=[[image] for image in batch_of_frames],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        smol_output_ids = smol_model.generate(
                **smol_inputs,
                max_new_tokens=200  ,
                do_sample=False,
            )
    input_len = smol_inputs["input_ids"].shape[1]
    captions = [
        smol_processor.decode(smol_output_ids[i, input_len:], skip_special_tokens=True)
        for i in range(len(batch_of_frames))
    ]
    if device == "cuda":
        del smol_inputs
        del smol_output_ids

    return captions
def generate_captions_in_batches_blip(batch_of_frames,
                                blip_model,
                                blip_processor,
                                device):
    blip_inputs = blip_processor(images = batch_of_frames, return_tensors = 'pt').to(device)
    with torch.no_grad():
        blip_output_ids = blip_model.generate(**blip_inputs,
                                        # max_length = 300, # run on 6gb vram
                                        min_length = 15,
                                        # no_repeat_ngram_size=3,
                                        # repetition_penalty=1.5,
                                        # early_stopping=True,
                                        # do_sample=False,
                                        num_beams = 3,
                                        )
        captions = blip_processor.batch_decode(blip_output_ids, skip_special_tokens=True)

    if device == "cuda":
        del blip_inputs
        del blip_output_ids

    return captions

def batched_captioning(frame_list: list,
                    batch_size: int,
                    smol_model,
                    smol_processor,
                    device):
    caption_list = []
    total_batches = (len(frame_list) + batch_size - 1) // batch_size
    with ui.make_progress() as progress:
        task_id = progress.add_task("  Analyzing frames", total=total_batches)
        for i in range(0, len(frame_list), batch_size):
            batch = frame_list[i:i + batch_size]
            caption = generate_captions_in_batches(batch,
                                                              smol_model=smol_model,
                                                              smol_processor=smol_processor,
                                                              device=device)
            caption_list.extend(caption)
            progress.advance(task_id)
    return caption_list


def batched_captioning_blip(frame_list: list,
                    batch_size: int,
                    blip_model,
                    blip_processor,
                    device):
    caption_list = []
    total_batches = (len(frame_list) + batch_size - 1) // batch_size
    with ui.make_progress() as progress:
        task_id = progress.add_task("  Analyzing frames", total=total_batches)
        for i in range(0, len(frame_list), batch_size):
            batch = frame_list[i:i + batch_size]
            caption = generate_captions_in_batches_blip(batch,
                                                                   blip_model=blip_model,
                                                                   blip_processor=blip_processor,
                                                                   device=device)
            caption_list.extend(caption)
            progress.advance(task_id)
    return caption_list

def caption_frame_collection(results_reranked: Dict[str, Any]) -> list:
    """Return caption records with their timing, not bare strings.

    The timestamps are what let the caller interleave frames with transcript
    chunks into one chronological timeline, so they must survive this hop.
    """
    frames = []
    for metadata in results_reranked["metadatas"][0]:
        if metadata.get("caption"):
            frames.append({
                "kind": "frame",
                "caption": metadata["caption"],
                "start_sec": metadata.get("start_sec"),
                "end_sec": metadata.get("end_sec"),
                "score": metadata.get("relevance_score", 0.0),
                "captioner": metadata.get("captioner"),
            })
    return frames

def generate_caption(frame,buffer, model: str):
    frame.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    response = chat(
    model=model,
    messages=[
        {
    "role": "system",
    "content": VLM_PROMPT
    },

    {
        'role': 'user',
        'content': '',
        'images': [img_bytes],
    }
    ],
    options={
            "num_predict": 200  
        }
    )
    return response.message.content

def rerank(query: str,
        results,
        reranker_model,
        top_k: int = 10) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    if not results['documents'][0]:
        return results
    captions = []
    metadata = results['metadatas'][0]
    
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for dic in metadata:
        for key, val in dic.items():
            if key == "caption":
                captions.append(val)
    pairs = [[query, caption] for caption in captions]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(captions, metadatas, distances, scores), key=lambda x: x[3], reverse=True)

    final_cap = []
    final_metas = []
    final_dists = []

    for cap, meta, dist, score in ranked[:top_k]:
        meta['relevance_score'] = float(score)
        final_cap.append(cap)
        final_metas.append(meta)
        final_dists.append(dist)

    return {
        'captions': [final_cap],
        'metadatas': [final_metas],
        'distances': [final_dists]
    }
    
def rerank_docs(query: str,
                results,
                reranker_model,
                top_k: int = 10) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    if not results['documents'][0]:
        return results

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    pairs = [[query, doc] for doc in documents]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(documents, metadatas, distances, scores), key=lambda x: x[3], reverse=True)

    final_docs = []
    final_metas = []
    final_dists = []

    for doc, meta, dist, score in ranked[:top_k]:
        meta['relevance_score'] = float(score)
        final_docs.append(doc)
        final_metas.append(meta)
        final_dists.append(dist)

    return {
        'documents': [final_docs],
        'metadatas': [final_metas],
        'distances': [final_dists]
    }    

def captioning_ollama(frame_PIL, model_name: str):
    """Caption frames via a local ollama VLM. Embedding happens in the caller."""
    captions = []
    with ui.make_progress() as progress:
        task_id = progress.add_task("  Processing frames", total=len(frame_PIL))
        for frame_no in frame_PIL:
            buffer = BytesIO()
            captions.append(generate_caption(frame_no, buffer, model_name))
            progress.advance(task_id)
    return captions


def _resize_long_edge(img: Image.Image, target: int = 896) -> Image.Image:
    """Scale so the long edge is `target`, preserving aspect ratio.

    Gemma 3's vision encoder works at 896x896. Squashing a 16:9 slide into a
    square throws away half the horizontal detail and distorts every glyph,
    which is what makes on-screen text unreadable to the captioner. Never
    upscale — that only adds interpolation noise.
    """
    w, h = img.size
    scale = target / max(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def temporal_persistence_filter(video_path: Path,
                                fps_selected: int = 2,
                                stable_fps: float = 0.5,
                                hash_threshold: int = 12,
                                persistence_window_sec: float = 5.0,
                                vote_ratio: float = 0.6,
                                # min_interval_sec: float = 2.0
                                ):
    """
    Two-pass pipeline:

    Pass 1 — State machine over tiny 9x8 gray frames piped from ffmpeg.
        Pipe delivers fps_selected (2fps) frames as 72-byte gray images.
        No full-resolution frames are decoded during this pass.

        STABLE state  → adaptive: only process every stable_step-th frame
                        (effective stable_fps = 0.5fps by default).
                        Remaining pipe bytes are drained but skipped.
                        Switches to per-frame processing on first hash spike.

        CANDIDATE state → process every pipe frame (full fps_selected rate).
                          Collect distances for persistence_window_sec.
                          CONFIRMED  (ratio > vote_ratio): record timestamp.
                          FALSE POSITIVE: discard, return to STABLE.

    Why adaptive sampling works here:
        During STABLE, the slide is static — checking every 2 seconds is
        enough to catch a change within one stable_step window (~2s lag).
        Once a spike is detected (CANDIDATE), we need full 2fps resolution
        to correctly time the persistence window and vote on distances.
        Switching back to STABLE resets to slow scanning immediately.

    Pass 2 — Targeted cv2 seeks for confirmed timestamps only.
        Only the confirmed slides (~20-50 per hour) get a full-res read.
        All other frames are never decoded at full resolution.

    Parameters:
        fps_selected          : pipe output rate (max sampling rate, used in CANDIDATE)
        stable_fps            : effective check rate in STABLE state (must be <= fps_selected)
        hash_threshold        : Hamming distance to consider "changed" (0=identical, 64=totally different)
        persistence_window_sec: how long a change must persist before it's trusted
        vote_ratio            : fraction of window frames that must stay above threshold
        min_interval_sec      : minimum gap between two selected frames
        max_interval_sec      : if no change detected for this long, force-select a frame

    Returns:
        frame_PIL   : list of PIL images (confirmed frames only)
        timestamps  : list of timestamp_sec for each selected frame
        ids         : list of unique frame ID strings
        fps         : video frame rate
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = total_frames / fps
    n_sampled = max(1, int(duration_sec * fps_selected))
    stable_step = max(1, round(fps_selected / stable_fps))
    HASH_W, HASH_H = 9, 8
    HASH_FRAME_SIZE = HASH_W * HASH_H 
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [
            ffmpeg_exe,
            "-threads", "0",
            "-skip_frame", "noref",
            "-i", str(video_path),
            "-vf", f"fps={fps_selected},scale={HASH_W}:{HASH_H}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "pipe:1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    STABLE, CANDIDATE = "STABLE", "CANDIDATE"
    state = STABLE
    ref_hash = None
    candidate = None       
    window_start = None
    last_selected_sec = None
    
    window_distances = []
    confirmed_timestamps = []
    frame_index = 0
    with ui.make_progress() as progress:
        task_id = progress.add_task("  Scanning video", total=n_sampled)
        while True:
            raw = proc.stdout.read(HASH_FRAME_SIZE)
            if len(raw) < HASH_FRAME_SIZE:
                break

            timestamp = frame_index / fps_selected
            frame_index += 1
            progress.advance(task_id)
            small = np.frombuffer(raw, dtype=np.uint8).reshape(HASH_H, HASH_W)
            current_hash = (small[:, 1:] > small[:, :-1]).flatten()
            if ref_hash is None:
                ref_hash = current_hash
                last_selected_sec = timestamp
                continue
            if state == STABLE and (frame_index % stable_step) != 0:
                continue
            distance = int(np.count_nonzero(current_hash ^ ref_hash))
            
            if state == STABLE:
                if distance > hash_threshold:
                    state = CANDIDATE
                    candidate = (timestamp, current_hash)
                    window_start = timestamp
                    window_distances = [distance]

            elif state == CANDIDATE:
                window_distances.append(distance)
                elapsed = timestamp - window_start
                if elapsed >= persistence_window_sec:
                    ratio = sum(d > hash_threshold for d in window_distances) / len(window_distances)
                    # time_ok = candidate[0] - last_selected_sec >= min_interval_sec
                    # if ratio > vote_ratio and time_ok:
                    if ratio > vote_ratio:
                        cand_ts, cand_hash = candidate
                        confirmed_timestamps.append(cand_ts)
                        ref_hash = cand_hash
                        last_selected_sec = cand_ts
                    state = STABLE
                    candidate = None
                    window_distances = []
    proc.stdout.close()
    proc.wait()

    # cv2's bundled ffmpeg can't software-decode AV1 (it only registers a
    # hwaccel path, which fails on platforms with no AV1 hw decoder and
    # returns no frame at all). imageio_ffmpeg's ffmpeg has libdav1d, so
    # frames are pulled through that binary instead of cap.read().
    frame_PIL = []
    valid_timestamps = []
    for ts in confirmed_timestamps:
        frame_proc = subprocess.run(
            [ffmpeg_exe, "-ss", str(ts), "-i", str(video_path),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        if frame_proc.returncode != 0 or not frame_proc.stdout:
            continue
        rgb = Image.open(BytesIO(frame_proc.stdout)).convert("RGB")
        gray = float(np.array(rgb.convert("L")).std())
        if gray < 5.0:
            continue
        frame_PIL.append(_resize_long_edge(rgb))
        valid_timestamps.append(ts)
    # ids = [f"{video_path}:{t}:Persistence_Filter" for t in valid_timestamps]
    # ui.info(f"Selected {len(frame_PIL)} keyframes from {total_frames} total frames")
    # return frame_PIL, valid_timestamps, ids, fps
    end_times = valid_timestamps[1:] + [duration_sec]

    ids = [f"{video_path}:{t}:Persistence_Filter" for t in valid_timestamps]
    ui.info(f"Selected {len(frame_PIL)} keyframes from {total_frames} total frames")
    return frame_PIL, valid_timestamps, end_times, ids, fps

   