"""
Order-preserving bridge for external captioning.

Stage 1 (export):  keyframes -> temp folder of PNGs + manifest.json
Stage 2 (external): you caption the PNGs with whatever model you like,
                    writing captions.json  (or filling the manifest's
                    "caption" fields)
Stage 3 (import):  manifest + captions -> caption list aligned 1:1 with
                   the original frame ordering, ready for the vector DB.

Nothing here imports from or modifies the gurrt package.
"""

import os
from pathlib import Path
import json
import shutil


MANIFEST_NAME = "manifest.json"


def export_keyframes(frame_PIL,
                     timestamps,
                     end_times,
                     ids,
                     out_dir: Path,
                     fps: float,
                     video_path=None,
                     overwrite: bool = False) -> Path:
    """
    Write frames as zero-padded PNGs plus a manifest recording the exact
    ordering and all per-frame metadata.

    Returns the manifest path.
    """
    out_dir = Path(out_dir)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(frame_PIL)
    # Fail loudly here rather than silently misaligning later.
    if not (len(timestamps) == len(end_times) == len(ids) == n):
        raise ValueError(
            f"length mismatch: frames={n} timestamps={len(timestamps)} "
            f"end_times={len(end_times)} ids={len(ids)}"
        )

    pad = max(4, len(str(n)))          # frame_0001.png -> lexical == numeric
    entries = []
    key_frame_path = out_dir / "key_frames"
    if key_frame_path.exists():
            shutil.rmtree(out_dir)
    key_frame_path.mkdir(parents=True, exist_ok=True)
    
    for i, (img, ts, ets, fid) in enumerate(zip(frame_PIL, timestamps, end_times, ids)):
        fname = f"frame_{i:0{pad}d}.png"
        img.save(key_frame_path/ fname)
        entries.append({
            "index": i,               # authoritative ordering
            "filename": fname,
            "id": fid,                # join key for reassembly
            "timestamp": ts,
            "end_time": ets,
            "caption": None,          # to be filled externally
        })

    manifest = {
        "video_path": str(video_path) if video_path is not None else None,
        "fps": fps,
        "count": n,
        "frames": entries,
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_captions(manifest_path: Path,
                  captions_path: Path ,
                  strict: bool = True):
    """
    Rebuild the caption list in the ORIGINAL frame order.

    captions_path may be:
      - None            -> captions read from the manifest's "caption" fields
      - a JSON dict     -> {filename_or_id: caption}
      - a JSON list     -> positional, must match count exactly

    strict=True raises on any missing caption; strict=False substitutes "".

    Returns (captions, timestamps, end_times, ids) all aligned by index.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    frames = sorted(manifest["frames"], key=lambda e: e["index"])   # never trust file order
    fps = manifest["fps"]
    lookup = {}
    if captions_path is not None:
        data = json.loads(Path(captions_path).read_text())
        if isinstance(data, dict):
            lookup = data
        elif isinstance(data, list):
            if len(data) != len(frames):
                raise ValueError(
                    f"positional caption list has {len(data)} entries, "
                    f"expected {len(frames)} — refusing to align by position"
                )
            lookup = {e["id"]: c for e, c in zip(frames, data)}
        else:
            raise TypeError("captions file must be a dict or list")

    captions, timestamps, end_times, ids = [], [], [], []
    missing = []
    for e in frames:
        if lookup:
            # accept keying by id OR by filename, whichever the tool emitted
            cap = lookup.get(e["id"], lookup.get(e["filename"]))
        else:
            cap = e.get("caption")
        if cap is None or (isinstance(cap, str) and not cap.strip()):
            missing.append(e["filename"])
            cap = ""
        captions.append(cap)
        timestamps.append(e["timestamp"])
        end_times.append(e["end_time"])
        ids.append(e["id"])

    if missing and strict:
        raise ValueError(
            f"{len(missing)} frame(s) have no caption: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}"
        )
    return captions, timestamps, end_times, ids, fps


def reload_frames(manifest_path: Path):
    """
    Optional: re-open the exported PNGs as PIL images in original order,
    for when the vector-DB step needs the images themselves and you don't
    want to re-run the extraction.
    """
    from PIL import Image
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    out_dir = manifest_path.parent
    frames = sorted(manifest["frames"], key=lambda e: e["index"])
    return [Image.open(out_dir /"key_frames" / e["filename"]).copy() for e in frames]
