from pathlib import Path
from gurrt.config.config import Settings
from gurrt.core.asr import audio_extract_chunk_and_embed
from gurrt.core.embedding import scene_detection_frame_sampling, scene_detection_frame_sampling_ollama
from gurrt.core.llm import LLMService
from gurrt.core.models import ModelManager
from gurrt.core.search import SearchService
from gurrt.core.vectordb import VectorDB

class VideoRag:
    def __init__(self, reset:bool = False):
        self.reset =reset
        self.settings = Settings()
        self.models = ModelManager(self.settings)
        self.vectordb = VectorDB(str(self.settings.CHROMA_DB_PATH), reset=reset)
        self.llm = LLMService(self.settings)
        self.device = self.models.device
        self.clip_model, self.clip_processor= self.models.get_clip()
        self.search = SearchService(clip_model=self.clip_model,
                                    clip_processor=self.clip_processor,
                                    settings= self.settings)
        
    def index_video(self, video_path:Path):
        if self.reset:
            try:
                w = self.llm.delete()
                if w:
                    print("\033[1;32mSupermemory Cleared\033[0m")
                else:
                    print("\033[1;32mSupermemory Not Cleared\033[0m")
            except:
                print("\033[1;32mSupermemory Initialized✅\033[0m")
        blip_model, blip_processor = self.models.get_blip()
        
        embeddings, metadatas, ids = scene_detection_frame_sampling(video_path= video_path,
                                                                    clip_model=self.clip_model,
                                                                    clip_processor=self.clip_processor,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    device = self.device)
        self.vectordb.add_frames(ids=ids,
                                 embeddings=embeddings,
                                 metadata=metadatas)
        self.models.release_blip()
        
        
    def index_video_ollama(self, video_path:Path, model_name: str):
        if self.reset:
            try:
                w = self.llm.delete()
                if w:
                    print("\033[1;32mSupermemory Cleared\033[0m")
                else:
                    print("\033[1;32mSupermemory Not Cleared\033[0m")
            except:
                print("\033[1;32mSupermemory Initialized✅\033[0m")
        embeddings, metadatas, ids = scene_detection_frame_sampling_ollama(
                                                                    video_path= video_path,
                                                                    clip_model=self.clip_model,
                                                                    clip_processor=self.clip_processor,
                                                                    model_name=model_name,
                                                                    device= self.device)
        self.vectordb.add_frames(ids=ids,
                                 embeddings=embeddings,
                                 metadata=metadatas)

    def index_audio(self, video_path:Path):
        whisper_model = self.models.get_whisper()
        chunked_text, metadatas, embeddings, ids = audio_extract_chunk_and_embed(
                                                                video_path=video_path,
                                                                clip_model=self.clip_model,
                                                                clip_processor=self.clip_processor,
                                                                whisper_model=whisper_model,
                                                                device = self.device)
        self.vectordb.add_asr(ids=ids,
                              embeddings=embeddings,
                              metadata=metadatas,
                              documents= chunked_text)
        self.models.release_whisper()
        
    async def ask(self, query:str):
        caption_list, asr_list = self.search.query_collection(self.device,
                                                              query,
                                                              n_results=10,
                                                              )
        result = await self.llm.query_llm(query, 
                                          caption_list=caption_list, 
                                          asr_list=asr_list)
        return result