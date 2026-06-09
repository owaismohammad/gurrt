from pathlib import Path
import numpy as np
from typing import Dict, Any
from ollama import chat
# import moviepy.editor as mp
from sentence_transformers import CrossEncoder
import torch
import cv2
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import subprocess
import imageio_ffmpeg  

# import imagehash
# from scenedetect import open_video, SceneManager
# from scenedetect.detectors import ContentDetector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gurrt.config.config import Settings




# def audio_extraction(path: Path):
#     settings = Settings()
#     audio_file = settings.AUDIO_PATH
#     video = mp.VideoFileClip(path)
#     audio = video.audio
#     audio.write_audiofile(audio_file)
    
#     audio.close()
#     video.close()
#     return audio_file

def audio_extraction(path: Path):
    settings = Settings()
    audio_file = settings.AUDIO_PATH
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_exe, "-y", "-i", str(path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(audio_file)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_file

def audio_to_text(audio_path, model, beam_size : int = 5) -> str:
    # segments, info = model.transcribe(audio_path, beam_size= beam_size, vad_filter = True)
    segments, info = model.transcribe(audio_path, batch_size=8, vad_filter=True)
    segments = list(segments)
    # text = ""
    # for segment in segments:
    #     text += segments.text
    # return text
    text = "".join(segment.text for segment in segments)
    return text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 40
    )
    chunked_text = text_splitter.split_text(text=text)
    return chunked_text

def scene_split(video_path):
    print("--- Detecting shot boundaries with PySceneDetect ---")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    try:
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
    except Exception as e:
        print("Scene detection failed:", e)
        scene_list = []
    return scene_list

def frame_listing_uniform(video_path: Path):
    cap= cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = range(0, total_frames, int(fps))

    frame_PIL = []
    timestamps_list = []
    ids = []
    with tqdm(total = len(frame_idx), desc="\033[1;32mProcessing frames...\033[0m") as pbar: 
        for frame_no in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(frame)
            frame_id = f"{video_path}:{timestamp_sec}:Uniform"
            
            frame_PIL.append(image)
            timestamps_list.append(timestamp_sec)
            ids.append(frame_id)
            pbar.update(1)
        cap.release()
    return frame_PIL, timestamps_list, ids, fps

def generate_captions_in_batches(batch_of_frames, 
                                 clip_model, 
                                 clip_processor, 
                                 blip_model, 
                                 blip_processor, 
                                 device):
    clip_inputs = clip_processor(images=batch_of_frames, return_tensors="pt").to(device)
    blip_inputs = blip_processor(images = batch_of_frames, return_tensors = 'pt').to(device)
    with torch.no_grad():
        clip_outputs = clip_model.get_image_features(clip_inputs.pixel_values)
        clip_outputs = clip_outputs.pooler_output
        clip_embeddings = clip_outputs / clip_outputs.norm(p=2, dim=-1, keepdim=True)
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

    embeddings_list = clip_embeddings.cpu().numpy().tolist()
    if device == "cuda":
        del clip_inputs
        del blip_inputs
        del clip_embeddings
        del blip_output_ids

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
   

    return captions, embeddings_list

def frame_listing(scene_list, video_path: Path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)   
    ids = []
    frame_PIL = []
    timestamps_list = []
    
    with tqdm(total = len(scene_list), desc="\033[1;32mProcessing frames...\033[0m") as pbar: 
        for i, scene in enumerate(scene_list):
            start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
            mid_time = (start_time + end_time) / 2
            timestamps = [start_time, mid_time, end_time]
            labels = ["initial", "middle", "final"]
            
            for t, label in zip(timestamps, labels):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                
                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(img_rgb)
                    frame_PIL.append(image)
                    timestamps_list.append(t)
                    frame_id = f"{video_path}:{t}:{label}"
                    ids.append(frame_id)
            pbar.update(1)
    return frame_PIL, timestamps_list, ids, fps

def batched_captioning(frame_list: list,
                       batch_size: int,
                       clip_model,
                       clip_processor,
                       blip_model,
                       blip_processor,
                       device):
    caption_list = []
    embedding_list = []
    
    with tqdm(total = (len(frame_list) + batch_size -1 ) // batch_size,
              desc = "\033[1;32mAnalyzing video visuals...\033[0m") as pbar:
        for i in range(0, len(frame_list),batch_size):
            batch = frame_list[i:i+batch_size]
            caption , embedding = generate_captions_in_batches(batch, 
                                                            clip_model= clip_model,
                                                            clip_processor= clip_processor,
                                                            blip_model= blip_model,
                                                            blip_processor= blip_processor,
                                                            device= device)
            caption_list.extend(caption)
            embedding_list.extend(embedding)
            pbar.update(1)
    return caption_list, embedding_list           


def caption_frame_collection(results_reranked: Dict[str, Any]) -> list:
    caption_list = []
    metadatas = results_reranked["metadatas"][0]
    for i, metadata in enumerate(metadatas):
        if metadata["caption"]:
            caption_list.append(metadata["caption"])
                
    return caption_list

def generate_caption(frame,buffer, model: str):
    frame.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    response = chat(
    model=model,
    messages=[
        {
    "role": "system",
    "content": "You are a helpful assistant that can analyze images and provide captions."
    },

    {
        'role': 'user',
        'content': 'What is in this image? .',
        'images': [img_bytes],
    }
    ],
    )

    return response.message.content



def rerank(query: str,
           results,
        #    MODEL_DIR:str,
        #    device,
           reranker_model,
           top_k: int = 10) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    # reranker_model = CrossEncoder(f"{MODEL_DIR}/reranker_model", device=device) 
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
                # MODEL_DIR: str,
                # device,
                reranker_model,
                top_k: int = 10) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    # reranker_model = CrossEncoder(f"{MODEL_DIR}/reranker_model", device=device)
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
def uniform_frame_sampling(path: Path,
                           clip_model,
                           clip_processor,
                           blip_processor,
                           blip_model,
                           device):
    cap= cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = range(0, total_frames, int(fps))
    # frame_no = 0
    embeddings = []
    metadatas = []
    ids = []
    # buffer = BytesIO()
    with tqdm(total = len(frame_idx), desc="\033[1;32mProcessing frames...\033[0m") as pbar: 
        for frame_no in frame_idx:
        # while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            # if frame_no % int(fps) == 0:
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
            # caption=generate_caption(image,buffer)
            # print(caption)
            blip_input = blip_processor(images = image,
                                        text="Describe the scene in a factual, objective manner.",
                                        return_tensors = 'pt').to(device)
            with torch.no_grad():
                outputs = clip_model.get_image_features(inputs.pixel_values)
                blip_outputs = blip_model.generate(**blip_input,
                                                #    max_length = 60,
                                                   min_length = 15,
                                                #    no_repeat_ngram_size=2,
                                                   num_beams = 3,
                                                    )
            
            caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
            image_embedding = outputs.pooler_output
            image_embedding = image_embedding / image_embedding.norm(dim = -1, keepdim= True)
            image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
            frame_id = f"{path}:{timestamp_sec}"
            
            ids.append(frame_id)
            embeddings.append(image_embedding)

            metadatas.append({
                "frame_idx": frame_no,
                "caption": caption,
                "timestamp_ms": timestamp_sec,
                "fps": fps,
                "source_path": str(path)
            })
            
            # frame_no +=1
            pbar.update(1)
    return embeddings, metadatas, ids

def uniform_frame_sampling_ollama(video_path: Path,
                                  model_name: str,
                                  clip_model,
                                  clip_processor, 
                                  device):
    cap= cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_idx = range(0, total_frames, int(fps))
    
    # frame_no = 0
    embeddings = []
    metadatas = []
    ids = []
    
    with tqdm(total = len(frames_idx), desc="\033[1;32mProcessing frames...\033[0m") as pbar: 
        for frame_no in frames_idx:
        # while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            # if frame_no % int(fps) == 0:
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
            buffer = BytesIO()
            caption=generate_caption(image,buffer, model_name)
            
            with torch.no_grad():
                outputs = clip_model.get_image_features(inputs.pixel_values)
            image_embedding = outputs.pooler_output
            image_embedding = image_embedding / image_embedding.norm(dim = -1, keepdim= True)
            image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
            frame_id = f"{video_path}:{timestamp_sec}"
            
            ids.append(frame_id)
            embeddings.append(image_embedding)

            metadatas.append({
                "frame_idx": frame_no,
                "caption": caption,
                "timestamp_ms": timestamp_sec,
                "fps": fps,
                "source_path": str(video_path)
            })
            
            # frame_no +=1
            pbar.update(1)
    return embeddings, metadatas, ids
def detect_scenes(video_path,
                  scene_list,
                  clip_processor,
                  clip_model,
                  model,
                  device):
    cap = cv2.VideoCapture(video_path)   
    embeddings = []
    metadatas = []
    ids = []
    
    with tqdm(total = len(scene_list), desc = "Processing frames") as pbar:
        for i, scene in enumerate(scene_list):
            start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
            mid_time = (start_time + end_time) / 2
            timestamps = mid_time
            labels = "middle"
            
            buffer = BytesIO()
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamps * 1000)
            ret, frame = cap.read()
            
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                caption=generate_caption(buffer=buffer,
                                         frame = image,
                                         model = model)
                inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
                with torch.no_grad():
                    outputs =clip_model.get_image_features(inputs.pixel_values)
                image_embedding = outputs.pooler_output
                image_embedding = image_embedding / image_embedding.norm(dim = -1, keepdim= True)
                image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
                timestamp_sec = timestamps*1000
                frame_id = f"{video_path}:{timestamp_sec}"
            
                ids.append(frame_id)
                embeddings.append(image_embedding)

                metadatas.append({
                    "frame_idx": f"frame_no_{i}_{labels}",
                    "caption": caption,
                    "timestamp_ms": timestamp_sec,
                    "source_path": str(video_path)
                })
            pbar.update(1)
               
    return embeddings, metadatas, ids

# Temporal Filtering with adaptive sampling and opening the video at full res -> 9x8 -> gray -> hash calc -> no video reseek
# def temporal_persistence_filter(video_path: Path,
#                                 fps_selected: int = 2,
#                                 hash_threshold : int = 12,
#                                 persistence_window_sec : float = 5.0,
#                                 vote_ratio : float = 0.6,
#                                 max_interval_sec: float = 60.0,
#                                 stable_fps: float = 0.5,
#                                 min_interval_sec: float = 2.0):
#     """
#     Pass 2 — Persistence State Machine + Pass 3 — Re-read selected frames.

#     The state machine has three states:

#         STABLE:
#             Watching for a hash spike. Every frame is compared to
#             reference_hash (the last known stable slide state).
#             If Hamming distance > hash_threshold → move to CANDIDATE.

#         CANDIDATE:
#             A spike was detected. We don't trust it yet.
#             Collect distances for the next persistence_window_sec seconds.
#             Two outcomes after the window expires:

#               CONFIRMED  (>= vote_ratio of window frames still above threshold)
#                 → Real slide change. Select the candidate frame.
#                 → Update reference_hash to current hash (stable new state).

#               FALSE POSITIVE (majority of frames returned below threshold)
#                 → Speaker moved and walked back. Discard silently.
#                 → Keep old reference_hash.

#     Why this filters the speaker:
#         Speaker walks in front of slide → hash spikes → speaker walks away
#         → within the window, frames return below threshold → vote fails → discarded.

#         New slide appears → hash spikes → ALL subsequent frames in window
#         also show high distance (slide is still there) → vote passes → selected.

#     Parameters:
#         hash_threshold        : Hamming distance to consider "changed" (0=identical, 64=totally different)
#         fps_selected          : frames per second to sample from the video
#         persistence_window_sec: how long a change must persist before it's trusted
#         vote_ratio            : fraction of window frames that must stay above threshold
#         min_interval_sec      : minimum gap between two selected frames
#         max_interval_sec      : if no change detected for this long, force-select a frame

#     Returns:
#         frame_PIL   : list of PIL images (confirmed frames only)
#         timestamps  : list of timestamp_sec for each selected frame
#         ids         : list of unique frame ID strings
#         fps         : video frame rate
#     """
#     # cv2 used only for a one-time header read — no frame decoding
#     cap = cv2.VideoCapture(str(video_path))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     frame_size = width * height * 3          # bgr24: 3 bytes per pixel
#     n_sampled = max(1, int((total_frames / fps) * fps_selected))
#     stable_step = max(1, int(fps_selected/stable_fps))
#     ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
#     proc = subprocess.Popen(
#         [
#             ffmpeg_exe,
#             "-threads", "0",
#             "-skip_frame", "noref",
#             "-i", str(video_path),
#             "-vf", f"fps={fps_selected}",
#             "-f", "rawvideo", "-pix_fmt", "bgr24",
#             "pipe:1"
#         ],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.DEVNULL
#     )

#     STABLE, CANDIDATE = "STABLE", "CANDIDATE"
#     state = STABLE
#     ref_hash = None
#     candidate = None          # (frame_index, timestamp, hash_bits, raw_bgr copy)
#     window_start = None
#     last_selected_sec = None
#     window_distances = []
#     selected_frames = []      # (timestamp, raw_bgr copy)

#     frame_index = 0

#     with tqdm(total=n_sampled, desc="\033[1;32mTemporal Persistence Filtering Frames...\033[0m") as pbar:
#         while True:
#             raw = proc.stdout.read(frame_size)
#             if len(raw) < frame_size:
#                 break

#             # read-only view — no copy until we actually need to store the frame
#             frame_view = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
#             timestamp = frame_index / fps_selected
#             frame_index += 1

#             gray = cv2.cvtColor(frame_view, cv2.COLOR_BGR2GRAY)
#             small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
#             current_hash = (small[:, 1:] > small[:, :-1]).flatten()

#             if ref_hash is None:
#                 ref_hash = current_hash
#                 last_selected_sec = timestamp
#                 pbar.update(1)
#                 continue
#             if state == STABLE and (frame_index % stable_step) != 0:
#                 pbar.update(1)
#                 continue
#             # if timestamp - last_selected_sec > max_interval_sec:
#             #     selected_frames.append((timestamp, frame_view.copy()))
#             #     ref_hash = current_hash
#             #     last_selected_sec = timestamp
#             #     state = STABLE
#             #     window_distances = []
#             #     candidate = None
#             #     pbar.update(1)
#             #     continue

#             distance = int(np.count_nonzero(current_hash ^ ref_hash))

#             if state == STABLE:
#                 if distance > hash_threshold:
#                     state = CANDIDATE
#                     candidate = (frame_index - 1, timestamp, current_hash, frame_view.copy())
#                     window_start = timestamp
#                     window_distances = [distance]

#             elif state == CANDIDATE:
#                 window_distances.append(distance)
#                 elapsed_time = timestamp - window_start
#                 if elapsed_time >= persistence_window_sec:
#                     ratio = sum(d > hash_threshold for d in window_distances) / len(window_distances)
#                     time_ok = candidate[1] - last_selected_sec >= min_interval_sec
#                     if ratio > vote_ratio and time_ok:
#                         _, cand_ts, cand_hash, cand_raw = candidate
#                         selected_frames.append((cand_ts, cand_raw))
#                         ref_hash = cand_hash
#                         last_selected_sec = cand_ts
#                     state = STABLE
#                     candidate = None
#                     window_distances = []

#             pbar.update(1)

#     proc.stdout.close()
#     proc.wait()

#     timestamp_sec = [f[0] for f in selected_frames]
#     ids = [f"{video_path}:{t}:Persistence_Filter" for t in timestamp_sec]
#     frame_PIL = [
#         Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
#         for _, raw in selected_frames
#     ]
#     print(f"\033[1;32mTotal sampled frames selected: {len(selected_frames)}\033[0m")
#     print(f"\033[1;32mTotal frames : {total_frames}\033[0m")
#     return frame_PIL, timestamp_sec, ids, fps


# Temporal Filtering with adaptive sampling and opening the video at 9x8 gray res/  video reseek

def temporal_persistence_filter(video_path: Path,
                                fps_selected: int = 2,
                                stable_fps: float = 0.5,
                                hash_threshold: int = 12,
                                persistence_window_sec: float = 5.0,
                                vote_ratio: float = 0.6,
                                max_interval_sec: float = 60.0,
                                min_interval_sec: float = 2.0):
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
    candidate = None       # (timestamp, hash_bits) — no raw frame stored
    window_start = None
    last_selected_sec = None
    window_distances = []
    confirmed_timestamps = []

    frame_index = 0

    with tqdm(total=n_sampled, desc="\033[1;32mTemporal Persistence Filtering Frames...\033[0m") as pbar:
        while True:
            raw = proc.stdout.read(HASH_FRAME_SIZE)
            if len(raw) < HASH_FRAME_SIZE:
                break

            timestamp = frame_index / fps_selected
            frame_index += 1
            pbar.update(1)

            

            small = np.frombuffer(raw, dtype=np.uint8).reshape(HASH_H, HASH_W)
            current_hash = (small[:, 1:] > small[:, :-1]).flatten()

            if ref_hash is None:
                ref_hash = current_hash
                last_selected_sec = timestamp
                continue
            if state == STABLE and (frame_index % stable_step) != 0:
                continue
            # if timestamp - last_selected_sec > max_interval_sec:
            #     confirmed_timestamps.append(timestamp)
            #     ref_hash = current_hash
            #     last_selected_sec = timestamp
            #     state = STABLE
            #     window_distances = []
            #     candidate = None
            #     continue

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
                    time_ok = candidate[0] - last_selected_sec >= min_interval_sec
                    if ratio > vote_ratio and time_ok:
                        cand_ts, cand_hash = candidate
                        confirmed_timestamps.append(cand_ts)
                        ref_hash = cand_hash
                        last_selected_sec = cand_ts
                    state = STABLE
                    candidate = None
                    window_distances = []

    proc.stdout.close()
    proc.wait()

    # Pass 2: targeted full-res seek for each confirmed timestamp only
    cap = cv2.VideoCapture(str(video_path))
    frame_PIL = []
    valid_timestamps = []

    for ts in confirmed_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frame_PIL.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            valid_timestamps.append(ts)
    cap.release()

    ids = [f"{video_path}:{t}:Persistence_Filter" for t in valid_timestamps]
    print(f"\033[1;32mTotal slides selected: {len(frame_PIL)}\033[0m")
    print(f"\033[1;32mTotal frames : {total_frames}\033[0m")
    return frame_PIL, valid_timestamps, ids, fps

   