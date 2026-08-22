"""Local llama.cpp answering backend for the benchmark harness.

`gurrt.core.openrouter` sends the finished system/user pair to a cloud model.
This is the same contract against a llama-server running a GGUF on this
machine, so a benchmark can swap the answering model by pointing at a
different file instead of changing an API key.

The transport is the one already used for captioning in
`gurrt.utils.llama_server_utils`: llama-server's OpenAI-compatible
`/v1/chat/completions`, with `/health` polled until the weights are resident.
Two things differ, both because this server answers questions rather than
captioning frames — no `--mmproj` (the answering models are text-only), and a
separate port so it can never collide with the captioning server on 8080.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests

from gurrt.cli import ui
from gurrt.config.config import LlamaServerManager

# 8080 belongs to the captioning server, whose client hardcodes that port.
DEFAULT_PORT = 8081


class LlamaChatError(RuntimeError):
    pass


def find_answer_ggufs(models_dir: Path) -> list[Path]:
    """GGUFs in `models_dir` that could serve as an answering model.

    The vision projector is not a model you can chat with, and the captioning
    weights are already spoken for, so neither is offered as a candidate.
    """
    if not models_dir.exists():
        return []
    return sorted(p for p in models_dir.glob("*.gguf")
                  if "mmproj" not in p.name.lower())


class LlamaChatServer:
    """A llama-server process holding one GGUF, used as a chat backend.

    Used as a context manager so the process — and its several GB of VRAM —
    is released even when a benchmark run raises partway through.
    """

    def __init__(self,
                 gguf_path: Path,
                 port: int = DEFAULT_PORT,
                 server_bin: Path | None = None,
                 n_gpu_layers: int = 99,
                 ctx_size: int = 8192,
                 max_tokens: int = 1024,
                 temperature: float = 0.3,
                 request_timeout_sec: int = 300):
        self.gguf_path = Path(gguf_path)
        self.port = port
        self.server_bin = Path(server_bin or LlamaServerManager().server_bin)
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout_sec = request_timeout_sec
        self._process: subprocess.Popen | None = None

    @property
    def model_name(self) -> str:
        """What the results are attributed to — the GGUF's own filename."""
        return self.gguf_path.stem

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.server_bin.exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {self.server_bin} — "
                "run `gurrt init-llama` first.")
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF not found: {self.gguf_path}")

        cmd = [
            str(self.server_bin),
            "-m", str(self.gguf_path),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "-n", str(self.max_tokens),
            "--port", str(self.port),
        ]
        ui.step(f"Starting answering model {self.model_name} on port {self.port}...")
        self._process = subprocess.Popen(cmd,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        self._wait_for_health()
        ui.success(f"Answering model ready: {self.model_name}")

    def _wait_for_health(self, attempts: int = 80, delay: float = 1.5) -> None:
        """Poll /health until the weights are loaded.

        Large quants can take a minute to page in, and llama-server answers
        503 while that happens. A request sent too early fails in a way that
        looks like a bad prompt, so the wait is not optional.
        """
        for _ in range(attempts):
            # A server that died on startup (bad quant, not enough VRAM) will
            # never answer, so stop waiting the moment the process is gone.
            if self._process and self._process.poll() is not None:
                raise LlamaChatError(
                    f"llama-server exited with code {self._process.returncode} "
                    f"while loading {self.gguf_path.name}. Common causes: not "
                    "enough VRAM for this quant, or an incompatible GGUF.")
            try:
                if requests.get(f"{self.base_url}/health",
                                timeout=2).status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(delay)
        raise LlamaChatError(
            f"{self.gguf_path.name} did not become healthy on port {self.port} "
            f"within {int(attempts * delay)}s.")

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            ui.info(f"Stopped answering model {self.model_name}")

    def __enter__(self) -> "LlamaChatServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── inference ────────────────────────────────────────────────────────────

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send one system/user pair and return the reply.

        Same shape as `openrouter.chat`, including how failures surface: the
        server's own message is passed through rather than being flattened
        into a generic error, because with local weights the message is
        usually the diagnosis (context overflow, template mismatch).
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repeat_penalty": 1.15,
            "frequency_penalty": 0.3,
            "top_p": 0.9,
            "min_p": 0.05,
        }
        try:
            resp = requests.post(f"{self.base_url}/v1/chat/completions",
                                 json=payload,
                                 timeout=self.request_timeout_sec)
        except requests.exceptions.RequestException as e:
            raise LlamaChatError(f"Could not reach {self.base_url}: {e}")

        if resp.status_code != 200:
            raise LlamaChatError(
                f"llama-server returned HTTP {resp.status_code} for "
                f"{self.model_name}: {resp.text[:500]}")

        try:
            data = resp.json()
        except ValueError:
            raise LlamaChatError(f"Unreadable response: {resp.text[:500]}")

        choices = data.get("choices") or []
        if not choices:
            raise LlamaChatError(f"No choices in response: {str(data)[:500]}")

        content = (choices[0].get("message") or {}).get("content") or ""
        content = content.strip()
        if not content:
            finish = choices[0].get("finish_reason")
            raise LlamaChatError(f"Empty reply (finish_reason={finish}).")
        return content
