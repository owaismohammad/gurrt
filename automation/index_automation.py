"""Part 1 of the benchmark harness — index a video the way the CLI does.

This is a thin driver, not a reimplementation. It calls the same
`VideoRag` methods that `gurrt index-llama` and `gurrt index` call, in the
same order, with the same `reset=True` semantics, so what gets measured is
the shipping pipeline rather than a parallel copy of it that can drift.

Two captioning paths are selectable:

    llama  — Gemma 3 under a local llama-server   (VideoRag.index_video_llama_server)
    blip2  — BLIP-2 in-process via transformers   (VideoRag.index_video_blip)

Both are followed by the shared audio pass (`VideoRag.index_audio`), which
transcribes, chunks and embeds the speech.

What is added on top of the CLI, and only this:

  * the run fails loudly if a stage produced nothing (the CLI's llama path
    swallows its own exceptions, which can leave an empty index looking fine),
  * the per-video captions.json the pipeline already writes is copied next to
    the results, so a benchmark run is self-contained.

Usage:

    python automation/index_automation.py data/dp17.mp4
    python automation/index_automation.py data/dp17.mp4 --captioner blip2
    python automation/index_automation.py a.mp4 b.mp4 --out-dir runs/today
"""

from __future__ import annotations

import argparse
from ast import Dict
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from platformdirs import user_config_dir

from automation.llama_inference import llama_inference
from gurrt.cli import ui
from gurrt.config.config import LlamaServerManager
# Private, but the alternative is duplicating its path-hashing scheme, which
# would silently stop pointing at the real log directory the day it changes.
from gurrt.core.debuglog import _video_dir
from gurrt.core.pipeline import VideoRag
# from gurrt.config.benchmark_config import VIDEO_PATH, OUTPUT_PATH, MANIFEST_PATH, CAPTION_PATH, DEFAULT_OUT, BASE_DIR, BENCHMARKING_DIR

DEFAULT_OUT_DIR = Path("automation_out")
CAPTIONERS = ("llama", "blip2")

home = Path(user_config_dir("gurrt"))
def require_captioner_assets(captioner: str) -> None:
    """Fail before any video work if the chosen captioner is not installed.
    Checked up front on purpose: keyframe extraction on an hour-long lecture
    is minutes of work, and discovering a missing GGUF afterwards wastes all
    of it.
    """
    if captioner != "llama":
        return
    mgr = LlamaServerManager(config_dir = home)
    missing = [str(p) for p in (mgr.server_bin, mgr.llm_path, mgr.mmproj_path)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "llama.cpp captioning engine is not installed — run "
            "`gurrt init-llama` first.\nMissing:\n  " + "\n  ".join(missing))

def _collection_counts(rag: VideoRag) -> tuple[int, int]:
    return (rag.vectordb.caption_collection.count(),
            rag.vectordb.asr_collection.count())

def index_video(video_path: Path,
                captioner: str,
                out_dir_bench: Path) -> dict:
    """Index one video exactly as the CLI would, and report what landed.

    A fresh `VideoRag(reset=True)` per video is deliberate — it is what
    `_do_index_llama` and `_do_index` do, and the reset is what keeps one
    video's questions from retrieving another video's frames.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    require_captioner_assets(captioner)

    mgr = LlamaServerManager(config_dir = home)
    rag = VideoRag(reset=True)

    video_start = time.time()
    if captioner == "llama":
        rag.index_video_llama_server(video_path=video_path,
                                    server_bin=mgr.server_bin,
                                    models_dir=mgr.models_dir,
                                    out_dir_bench = out_dir_bench,
                                    max_workers = 64)
    elif captioner == "blip2":
        rag.index_video_blip(video_path=video_path,
                             out_dir = out_dir_bench)
    else:
        raise ValueError(f"Unknown captioner {captioner!r}, "
                         f"expected one of {CAPTIONERS}")
    video_sec = time.time() - video_start

    audio_start = time.time()
    rag.index_audio(video_path=video_path)
    audio_sec = time.time() - audio_start

    frame_count, asr_count = _collection_counts(rag)
    if frame_count == 0:
        raise RuntimeError(
            f"Captioning produced no frames for {video_path.name} — the "
            "index is empty. Check the captioning server's output above.")
    if asr_count == 0:
        ui.warn(f"No transcript chunks for {video_path.name} — answers will "
                "rest on captions alone.")

    manifest = {
        "video": str(video_path),
        "video_name": video_path.name,
        "captioner": captioner,
        "frames_indexed": frame_count,
        "transcript_chunks_indexed": asr_count,
        "timings_sec": {
            "video": round(video_sec, 1),
            "audio": round(audio_sec, 1),
            "total": round(video_sec + audio_sec, 1),
        },
    }

    # log_captions() already wrote the full caption + metadata JSON during the
    # run; copying it keeps the results folder self-contained.
    if out_dir_bench is not None:
        written = _video_dir(rag.settings, video_path) / "captions.json"
        if written.exists():
            target = out_dir_bench / f"{video_path.stem}.captions.json"
            shutil.copyfile(written, target)
            manifest["captions_json"] = str(target)

    ui.success(f"{video_path.name}: {frame_count} frames + {asr_count} "
               f"transcript chunks in {manifest['timings_sec']['total']}s "
               f"({captioner})")
    return manifest


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Index videos through the CLI's own indexing pipeline.")
    # p.add_argument("videos", nargs="+", type=Path,
    #                help="One or more video files to index.")
    # p.add_argument("--captioner", choices=CAPTIONERS, default="llama",
    #                help="Frame captioner to use (default: llama).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Where manifests are written (default: {DEFAULT_OUT_DIR}).")
    return p.parse_args(argv)
BASE_DIR = Path(__file__).resolve().parents[3] / "workspace"
print(BASE_DIR)



ROOT_BASE_DIR = Path(__file__).resolve().parents[2]
print(ROOT_BASE_DIR)
BENCHMARKING_DIR = ROOT_BASE_DIR / "Benchmarking - 2"
BENCHMARKING_DIR.mkdir(parents = True, exist_ok= True)

VIDEO_DIR = ROOT_BASE_DIR / "Videos"
VIDEO_DIR.mkdir(parents = True, exist_ok= True)

QUES_DIR = ROOT_BASE_DIR / "Questions"
QUES_DIR.mkdir(parents = True, exist_ok= True)
def main(argv=None):
    args = _parse_args(argv)
    for id in range(1, 21):
    
        video_path = VIDEO_DIR / f"Video_ID_{id}.mp4"
        
        Video_ID =  BENCHMARKING_DIR / f"Video_ID_{id}"
        Video_ID.mkdir(parents = True, exist_ok= True)
        
        print(f"Starting Video_ID_{id}")
        try:
            manifest = index_video(video_path=video_path,
                                captioner="llama",
                                out_dir_bench=Video_ID)
        except Exception as e:
            ui.error(f"Indexing failed: {e}")
            continue
        out = Video_ID / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
        ui.success(f"Indexed {len(manifest)} video(s) → {out}")
        
        llama_inference(questions_csv=QUES_DIR / f"Video_ID_{id}.csv",
                        output_dir=Video_ID,
                        max_workers=10)

if __name__ == "__main__":
    sys.exit(main())