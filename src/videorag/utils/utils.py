from pathlib import Path
from typing import Dict, Any
from ollama import chat
import moviepy.editor as mp
from sentence_transformers import CrossEncoder
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from videorag.config.config import Settings
# from dotenv import load_dotenv
# load_dotenv()

# MODEL_DIR = os.getenv("MODEL_CACHE_DIR")
# if MODEL_DIR is None:
#     raise RuntimeError("MODEL_CACHE_DIR is not set")

device = "cuda" if torch.cuda.is_available() else "cpu"



def audio_extraction(path: Path):
    settings = Settings()
    audio_file = settings.AUDIO_PATH
    video = mp.VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile(audio_file)
    
    audio.close()
    video.close()
    return audio_file

def audio_to_text(audio_path, model) -> str:
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = ""
    for segment in segments:
        text+=segment.text
    return text

def chunk_text(text):
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20
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

def generate_captions_in_batches(batch_of_frames, 
                                 clip_model, 
                                 clip_processor, 
                                 blip_model, 
                                 blip_processor, 
                                 device="cuda"):
    clip_inputs = clip_processor(images=batch_of_frames, return_tensors="pt").to(device)
    blip_inputs = blip_processor(images = batch_of_frames, return_tensors = 'pt').to(device)
    with torch.no_grad():
        clip_outputs = clip_model.get_image_features(clip_inputs.pixel_values)
        clip_outputs = clip_outputs.pooler_output
        clip_embeddings = clip_outputs / clip_outputs.norm(p=2, dim=-1, keepdim=True)
        blip_output_ids = blip_model.generate(**blip_inputs,
                                        # max_length = 300, # run on 6gb vram
                                        # min_length = 100,
                                        # no_repeat_ngram_size=3,
                                        # repetition_penalty=1.5,
                                        # early_stopping=True,
                                        # do_sample=False,
                                        # num_beams = 3,
                                        )
        captions = blip_processor.batch_decode(blip_output_ids, skip_special_tokens=True)

   
            
    embeddings_list = clip_embeddings.cpu().numpy().tolist()
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
    
    with tqdm(total = len(scene_list), desc="processing frames ") as pbar: 
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

def batched_captioning(frame_list: list, batch_size: int, clip_model, clip_processor, blip_model, blip_processor):
    caption_list = []
    embedding_list = []
    
    with tqdm(total = int(len(frame_list)/ batch_size), desc = "Batched image captioning") as pbar:
        for i in range(0, len(frame_list),batch_size):
            batch = frame_list[i:i+batch_size]
            caption , embedding = generate_captions_in_batches(batch, 
                                                            clip_model= clip_model,
                                                            clip_processor= clip_processor,
                                                            blip_model= blip_model,
                                                            blip_processor= blip_processor)
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

# def generate_caption(frame,buffer):
#     frame.save(buffer, format="JPEG")
#     img_bytes = buffer.getvalue()
#     response = chat(
#     model='gemma3',
#     messages=[
#         {
#     "role": "system",
#     "content": "You are a helpful assistant that can analyze images and provide captions."
#     },

#     {
#         'role': 'user',
#         'content': 'What is in this image? .',
#         'images': [img_bytes],
#     }
#     ],
#     )

#     return response.message.content



def rerank(query: str, results, MODEL_DIR:str, top_k: int = 10) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    reranker_model = CrossEncoder(f"{MODEL_DIR}/reranker_model", device=device) 
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
    
def rerank_docs(query: str, results, MODEL_DIR, top_k: int = 10) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    reranker_model = CrossEncoder(f"{MODEL_DIR}/reranker_model", device=device)
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
# def uniform_frame_sampling(path: str, clip_model, clip_processor, blip_processor, blip_model):
#     cap= cv2.VideoCapture(path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_no = 0
#     embeddings = []
#     metadatas = []
#     ids = []
#     # buffer = BytesIO()
#     with tqdm(total=total_frames, desc="Processing frames") as pbar:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_no % int(fps) == 0:
#                 timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(frame)
#                 inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
#                 # caption=generate_caption(image,buffer)
#                 # print(caption)
#                 blip_input = blip_processor(images = image,
#                                             text="Describe the scene in a factual, objective manner.",
#                                             return_tensors = 'pt').to(device)
#                 with torch.no_grad():
#                     outputs = clip_model.get_image_features(inputs.pixel_values)
#                     blip_outputs = blip_model.generate(**blip_input,
#                                                        max_length = 60,
#                                                        min_length = 20,
#                                                        no_repeat_ngram_size=2,
#                                                        num_beams = 5,
#                                                        )
                
#                 caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
#                 image_embedding = outputs.pooler_output
#                 image_embedding = image_embedding / image_embedding.norm(dim = -1, keepdim= True)
#                 image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
#                 frame_id = f"{path}:{timestamp_sec}"
                
#                 ids.append(frame_id)
#                 embeddings.append(image_embedding)

#                 metadatas.append({
#                     "frame_idx": frame_no,
#                     "caption": caption,
#                     "timestamp_ms": timestamp_sec,
#                     "fps": fps,
#                     "source_path": path
#                 })
                
#             frame_no +=1
#             pbar.update(1)
#     return embeddings, metadatas, ids