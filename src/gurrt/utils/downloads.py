"""Real progress for model downloads.

Weights run from hundreds of megabytes to several gigabytes. A spinner says
only that the process has not crashed; it cannot distinguish a stalled
connection from a slow one, and gives no way to decide whether to wait.

Two strategies, because the downloaders differ:

- For a plain URL we stream it ourselves and count bytes exactly.
- For huggingface_hub, which owns its own transfer and offers no progress
  callback, we run it on a worker thread and watch bytes land on disk. That
  works the same whether the caller uses hf_hub_download, snapshot_download
  or from_pretrained, without reaching into any of them.

Nothing here changes where files end up.
"""
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gurrt.cli import ui

_POLL_SEC = 0.25
_CHUNK = 1 << 20  # 1 MiB


def _dir_size(path: Path) -> int:
    """Bytes currently on disk under `path`, partial files included."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue          # file vanished mid-walk; it was a temp
    except OSError:
        pass
    return total


def hf_cache_dir_for(repo_id: str) -> Path:
    """Where huggingface_hub stores a repo, so we can watch it grow."""
    from huggingface_hub.constants import HF_HUB_CACHE
    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def hf_file_size(repo_id: str, filename: str):
    """Exact size of one repo file, or None if the API is unreachable."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        return get_hf_file_metadata(hf_hub_url(repo_id, filename)).size
    except Exception:
        return None


def hf_repo_size(repo_id: str):
    """Total size of a repo's weights, or None.

    Excludes the duplicate weight formats that from_pretrained will not
    fetch, so the total roughly matches what actually downloads.
    """
    try:
        from huggingface_hub import HfApi
        skip = (".h5", ".msgpack", ".ot", ".tflite", ".onnx")
        return sum(
            f.size or 0
            for f in HfApi().model_info(repo_id, files_metadata=True).siblings
            if not f.rfilename.endswith(skip)
        ) or None
    except Exception:
        return None


def watch_download(description: str, worker, watch_dir: Path, total_bytes=None):
    """Run `worker()` while reporting bytes appearing under `watch_dir`.

    Progress is measured as growth from the directory's starting size, so a
    partially-populated cache reports only what this call adds rather than
    jumping to a misleading near-complete bar.
    """
    watch_dir = Path(watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    baseline = _dir_size(watch_dir)

    with ui.make_download_progress() as progress:
        task = progress.add_task(description, total=total_bytes)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker)
            while not future.done():
                done = max(0, _dir_size(watch_dir) - baseline)
                if total_bytes:
                    done = min(done, total_bytes)
                progress.update(task, completed=done)
                time.sleep(_POLL_SEC)

            result = future.result()   # re-raises anything the worker hit
            progress.update(
                task,
                total=total_bytes or max(1, _dir_size(watch_dir) - baseline),
                completed=total_bytes or max(1, _dir_size(watch_dir) - baseline),
            )
    return result


def stream_download(url: str, dest: Path, description: str, headers=None) -> Path:
    """Download a URL to `dest`, counting bytes as they arrive.

    Writes to a .part file and renames on success, so an interrupted
    download never leaves something that looks like a usable file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length") or 0) or None
        with ui.make_download_progress() as progress, open(part, "wb") as out:
            task = progress.add_task(description, total=total)
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                progress.advance(task, len(chunk))
            if total is None:
                progress.update(task, total=out.tell(), completed=out.tell())

    part.replace(dest)
    return dest
