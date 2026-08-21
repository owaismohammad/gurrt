from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import requests

from gurrt.config.config import Settings
from gurrt.core.asr import audio_extract_chunk_and_embed
from gurrt.core.embedding import (
    frame_detection,
    frame_detection_blip,
    frame_detection_ollama,
    captioning_and_embedding_llama_server)

from gurrt.utils.llama_server_utils import process_video,wait_for_server
from gurrt.core.llm import LLMService, format_timeline
from gurrt.core.models import ModelManager
from gurrt.core.search import SearchService
from gurrt.core.debuglog import log_captions, log_transcript
from gurrt.core.vectordb import VectorDB
from gurrt.config.config import LlamaServerManager
from gurrt.cli import ui
import subprocess
import requests
import time

class VideoRag:
    def __init__(self, reset:bool = False):
        self.reset =reset
        self.settings = Settings()
        self.models = ModelManager(self.settings)
        self.vectordb = VectorDB(str(self.settings.CHROMA_DB_PATH), reset=reset)
        # self.llm = LLMService(self.settings)
        self.device = self.models.device
        self.text_embedder = self.models.get_text_embedder()

    def index_video(self, video_path:Path, flag:bool):
        if self.reset:
            try:
                self.llm.delete()
            except Exception:
                pass
        embeddings, metadatas, ids = frame_detection(video_path= video_path,
                                                    text_embedder=self.text_embedder,
                                                    models = self.models,
                                                    device = self.device,
                                                    flag = flag,
                                                    settings = self.settings)
        self.vectordb.add_frames(ids=ids,
                                embeddings=embeddings,
                                metadata=metadatas)
        log_captions(self.settings, video_path, metadatas, ids)
        self.models.release_smol()

    def index_video_blip(self, video_path:Path,
                         out_dir: Path):
        # if self.reset:
        #     try:
        #         self.llm.delete()
        #     except Exception:
        #         pass
        # frame_detection_blip(video_path= video_path,
        #                     # text_embedder=self.text_embedder,
        #                     # models = self.models,
        #                     # device = self.device,
        #                     # settings = self.settings,
        #                     out_dir = out_dir)
        embeddings, metadatas, ids = frame_detection_blip(video_path= video_path,
                                                    text_embedder=self.text_embedder,
                                                    models = self.models,
                                                    device = self.device,
                                                    settings = self.settings,
                                                    out_dir = out_dir)
        self.vectordb.add_frames(ids=ids,
                                embeddings=embeddings,
                                metadata=metadatas)
        log_captions(self.settings, video_path, metadatas, ids)
        # self.models.release_blip()

    def index_video_ollama(self, video_path:Path, model_name: str):
        if self.reset:
            try:
                self.llm.delete()
            except Exception:
                pass
        embeddings, metadatas, ids = frame_detection_ollama(
                                                            video_path= video_path,
                                                            text_embedder=self.text_embedder,
                                                            model_name=model_name,
                                                            device= self.device,
                                                            settings= self.settings)
        self.vectordb.add_frames(ids=ids,
                                embeddings=embeddings,
                                metadata=metadatas)
        log_captions(self.settings, video_path, metadatas, ids)


    def index_video_llama_server(self, video_path: Path, server_bin: Path, models_dir: Path):
        if self.reset:
            try:
                self.llm.delete()
            except Exception:
                pass
        llama_server_manager = LlamaServerManager()        
        cmd_caption_server = [
            str(server_bin),
            "-m", str(llama_server_manager.llm_path),
            "--mmproj", str(llama_server_manager.mmproj_path),
            "-ngl", "99",
            "--parallel", "4",
            "-c", "8192",
            "--port", "8080",
            "-n","320",
            #"--flash-attn"
        ]

        process_caption = None

        try:
            ui.step("Launching Gemma 3 captioning server on port 8080...")
            process_caption = subprocess.Popen(
                cmd_caption_server, 
                # stdout=subprocess.DEVNULL, 
                # stderr=subprocess.DEVNULL
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_server = executor.submit(wait_for_server)
                future_video = executor.submit(process_video, video_path)
                server_ready = future_server.result()
                frame_PIL, timestamps_list, end_times, ids, fps = future_video.result()

            if not server_ready:
                raise TimeoutError("Captioning engine failed to initialize within VRAM allocation limits.")

            ui.success("Video frames processed and captioning server ready")
            embeddings, metadatas, ids = captioning_and_embedding_llama_server(
                                                                        frame_PIL= frame_PIL,
                                                                        text_embedder=self.text_embedder,
                                                                        timestamps_list= timestamps_list,
                                                                        end_times= end_times,
                                                                        ids= ids,
                                                                        fps= fps,
                                                                        video_path= video_path,
                                                                        settings= self.settings)              
            self.vectordb.add_frames(ids=ids,
                                        embeddings=embeddings,
                                        metadata=metadatas)
            log_captions(self.settings, video_path, metadatas, ids)
        except Exception as e:
            ui.error(f"Pipeline failed: {e}")
        finally:
            if process_caption:
                process_caption.terminate()
                process_caption.wait()
    
    def index_audio(self, video_path:Path):
        whisper_model = self.models.get_whisper()
        chunked_text, metadatas, embeddings, ids = audio_extract_chunk_and_embed(
                                                            video_path=video_path,
                                                            settings = self.settings,
                                                            text_embedder=self.text_embedder,
                                                            whisper_model=whisper_model,
                                                            device = self.device)
        self.vectordb.add_asr(ids=ids,
                            embeddings=embeddings,
                            metadata=metadatas,
                            documents= chunked_text)
        log_transcript(self.settings, video_path, chunked_text, metadatas, ids)
        self.models.release_whisper()

    async def ask(self, query:str):
        reranker = self.models.get_reranker()
        
        search = SearchService(text_embedder=self.text_embedder,
                                reranker= reranker,
                                vectordb= self.vectordb,
                                settings= self.settings)
        caption_list, asr_list, stats = search.build_context(
            self.device,
            query,
            top_k=self.settings.ASK_TOP_K,
            candidate_k=self.settings.ASK_CANDIDATE_K)
        timeline = format_timeline(caption_list, asr_list)
        ui.info(f"Context: {len(caption_list)} frames + {len(asr_list)} "
                f"transcript chunks")
        if stats["low_fidelity_visual"]:
            ui.warn(f"Captions from {', '.join(stats['captioners'])} are "
                    "low-fidelity — weighting the transcript instead")
        result = await self.llm.query_llm(
            query,
            timeline=timeline,
            low_fidelity_visual=stats["low_fidelity_visual"])
        self.models.release_all()
        return result