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
import json
import shutil
import sys
import time
from pathlib import Path

from gurrt.cli import ui
from gurrt.config.config import LlamaServerManager
# Private, but the alternative is duplicating its path-hashing scheme, which
# would silently stop pointing at the real log directory the day it changes.
from gurrt.core.debuglog import _video_dir
from gurrt.core.pipeline import VideoRag
from gurrt.config.benchmark_config import VIDEO_PATH, OUTPUT_PATH, MANIFEST_PATH, CAPTION_PATH, DEFAULT_OUT

DEFAULT_OUT_DIR = Path("automation_out")
CAPTIONERS = ("llama", "blip2")


def require_captioner_assets(captioner: str) -> None:
    """Fail before any video work if the chosen captioner is not installed.

    Checked up front on purpose: keyframe extraction on an hour-long lecture
    is minutes of work, and discovering a missing GGUF afterwards wastes all
    of it.
    """
    if captioner != "llama":
        return
    mgr = LlamaServerManager()
    missing = [str(p) for p in (mgr.server_bin, mgr.llm_path, mgr.mmproj_path)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "llama.cpp captioning engine is not installed — run "
            "`gurrt init-llama` first.\nMissing:\n  " + "\n  ".join(missing))


def _collection_counts(rag: VideoRag) -> tuple[int, int]:
    return (rag.vectordb.caption_collection.count(),
            rag.vectordb.asr_collection.count())


def index_video(video_path: Path, captioner: str,
                out_dir: Path | None = None) -> dict:
    """Index one video exactly as the CLI would, and report what landed.

    A fresh `VideoRag(reset=True)` per video is deliberate — it is what
    `_do_index_llama` and `_do_index` do, and the reset is what keeps one
    video's questions from retrieving another video's frames.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    require_captioner_assets(captioner)

    mgr = LlamaServerManager()
    rag = VideoRag(reset=False)

    video_start = time.time()
    if captioner == "llama":
        rag.index_video_llama_server(video_path=video_path,
                                     server_bin=mgr.server_bin,
                                     models_dir=mgr.models_dir)
    elif captioner == "blip2":
        rag.index_video_blip(video_path=video_path,
                             out_dir = out_dir)
    else:
        raise ValueError(f"Unknown captioner {captioner!r}, "
                         f"expected one of {CAPTIONERS}")
    video_sec = time.time() - video_start

    audio_start = time.time()
    rag.index_audio(video_path=video_path)
    audio_sec = time.time() - audio_start

    # index_video_llama_server catches its own exceptions and only prints, so
    # a failed caption pass returns normally. Without this check the benchmark
    # would go on to score answers drawn from an empty index.
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
    if out_dir is not None:
        written = _video_dir(rag.settings, video_path) / "captions.json"
        if written.exists():
            target = out_dir / f"{video_path.stem}.captions.json"
            shutil.copyfile(written, target)
            manifest["captions_json"] = str(target)

    ui.success(f"{video_path.name}: {frame_count} frames + {asr_count} "
               f"transcript chunks in {manifest['timings_sec']['total']}s "
               f"({captioner})")
    return manifest


def index_videos(video_paths: list[Path], captioner: str,
                 out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    require_captioner_assets(captioner)
    for p in video_paths:
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")

    if len(video_paths) > 1:
        reset=True  #per video is the CLI's behaviour and the benchmark relies
        # on it, but it means each video wipes the previous one's index.
        ui.warn(f"{len(video_paths)} videos: each reset clears the previous "
                "index, so only the last stays queryable afterwards.")

    manifests = []
    for video_path in video_paths:
        ui.step(f"Indexing {video_path.name} with {captioner}")
        manifests.append(index_video(video_path, captioner, out_dir))
    return manifests


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


def main(argv=None) -> int:
    args = _parse_args(argv)
    captioner = "blip2"
    video_id = 1
    # for video_id in range(1, 2):
        
    video_path =VIDEO_PATH
    out_path =  OUTPUT_PATH
    manifests = index_videos(video_paths=[video_path], captioner="llama", out_dir=out_path)
    # try:
    #     manifests = index_videos(args.videos, args.captioner, args.out_dir)
    # except Exception as e:
    #     ui.error(f"Indexing failed: {e}")
    #     return 1

    out = args.out_dir / "manifest.json"
    out.write_text(json.dumps(manifests, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    ui.success(f"Indexed {len(manifests)} video(s) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())