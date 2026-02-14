from pathlib import Path
from videorag.config.config import Settings
from videorag.core.asr import audio_extract_chunk_and_embed
from videorag.core.embedding import scene_detection_frame_sampling
from videorag.core.llm import LLMService
from videorag.core.models import ModelManager
from videorag.core.search import SearchService
from videorag.core.vectordb import VectorDB

class VideoRag:
    def __init__(self):
        
        self.settings = Settings()
        self.models = ModelManager(self.settings)
        self.vectordb = VectorDB(str(self.settings.CHROMA_DB_PATH))
        self.llm = LLMService(self.settings)
        
        self.clip_model, self.clip_processor= self.models.get_clip()
        self.search = SearchService(clip_model=self.clip_model,
                                    clip_processor=self.clip_processor,
                                    settings= self.settings)
        
    def index_video(self, video_path:Path):
        
        blip_model, blip_processor = self.models.get_blip()
        embeddings, metadatas, ids = scene_detection_frame_sampling(video_path= video_path,
                                                                    clip_model=self.clip_model,
                                                                    clip_processor=self.clip_processor,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor)
        self.vectordb.add_frames(ids=ids,
                                 embeddings=embeddings,
                                 metadata=metadatas)
        self.models.release_blip()
        
    def index_audio(self, video_path:Path):
        whisper_model = self.models.get_whisper()
        chunked_text, metadatas, embeddings, ids = audio_extract_chunk_and_embed(
                                                                video_path=video_path,
                                                                clip_model=self.clip_model,
                                                                clip_processor=self.clip_processor,
                                                                whisper_model=whisper_model)
        self.vectordb.add_asr(ids=ids,
                              embeddings=embeddings,
                              metadata=metadatas,
                              documents= chunked_text)
        self.models.release_whisper()
        
    async def ask(self, query:str):
        caption_list, asr_list = self.search.query_collection(query,
                                                              n_results=10)
        result = await self.llm.query_llm(query, 
                                          caption_list=caption_list, 
                                          asr_list=asr_list)
        return result
    
    async def delete(self):
        await self.llm.delete()