"""Image-maker agent: black-forest-labs/flux.1-schnell via NVIDIA Cloud API.

Endpoint (from build.nvidia.com source):
  POST https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell
  Headers: Authorization: Bearer $NVIDIA_API_KEY, Accept: application/json,
           Content-Type: application/json
  Payload: {"prompt": str, "width": int, "height": int, "seed": int, "steps": int}
  Response: {"artifacts": [{"base64": "<png/jpg b64>", "seed": ..., "finishReason": ...}]}
  Decode artifacts[0].base64 -> .jpg file in ./outputs
"""
from __future__ import annotations

import base64
import os
import random
import re
import time

import config
from agency.agent_memory import AgentMemoryMixin
from agency.rate_limiter import limiter

INVOKE_URL = os.getenv(
    "IMAGE_URL",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
)
MODEL_ID = "black-forest-labs/flux.1-schnell"

# schnell is distilled for 1-4 steps; 4 = best quality
DEFAULT_STEPS = int(os.getenv("IMAGE_STEPS", "4"))
DEFAULT_SEED = 0  # 0 = random each call (server-side)
OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "./outputs")

# Supported resolutions per model card
SIZES = {
    "square": (1024, 1024),
    "portrait": (768, 1344),
    "landscape": (1344, 768),
    "tall": (768, 1344),
    "wide": (1344, 768),
}


def parse_size(text: str) -> tuple[int, int]:
    """Extract WxH or portrait/landscape/square keywords. Default 1024x1024."""
    t = (text or "").lower()
    m = re.search(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", t)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        # clamp to sane range
        return max(256, min(w, 1536)), max(256, min(h, 1536))
    for key, (w, h) in SIZES.items():
        if key in t:
            return w, h
    return 1024, 1024


def parse_steps(text: str) -> int:
    m = re.search(r"steps?\s*[:=]?\s*([1-8])", (text or "").lower())
    if m:
        return max(1, min(int(m.group(1)), 8))
    return DEFAULT_STEPS


def build_prompt(instruction: str, context: str = "") -> str:
    """Combine context + instruction into one image prompt, strip agent prefixes."""
    raw = f"{context}\n{instruction}".strip() if context else (instruction or "").strip()
    # drop leading task verbs so the diffusion prompt stays clean
    raw = re.sub(
        r"^(please\s+)?(generate|create|make|draw|render|produce)\s+(an?\s+)?(image|picture|photo|art|poster|logo)\s+(of|showing|with|for)?\s*",
        "",
        raw,
        flags=re.I,
    ).strip()
    # remove overall-goal prefix from agency context
    raw = re.sub(r"^overall goal:\s*", "", raw, flags=re.I).strip()
    return raw[:1500] or "a cinematic studio photo, high detail"


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 4,
    timeout: int = 120,
    retries: int = 3,
    on_retry=None,
) -> tuple[bytes, int]:
    """POST to FLUX schnell, return (image_bytes, seed_used). Retries transient errors."""
    import requests

    api_key = config.require_key()
    if seed == 0:
        seed = random.randint(0, 2**31 - 1)
    payload = {"prompt": prompt, "width": width, "height": height, "seed": seed, "steps": steps}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        limiter.wait()  # share budget with chat agents
        try:
            resp = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"FLUX {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            arts = data.get("artifacts")
            b64 = None
            used_seed = seed
            if isinstance(arts, list) and arts:
                b64 = arts[0].get("base64")
                used_seed = arts[0].get("seed", seed)
            elif isinstance(arts, dict):
                b64 = arts.get("base64")
                used_seed = arts.get("seed", seed)
            elif isinstance(data.get("image"), str):  # fallback shape
                b64 = data["image"]
            if not b64:
                raise RuntimeError(f"No image in response keys={list(data.keys())}")
            # strip data-uri prefix if present
            if "," in b64 and b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            return base64.b64decode(b64), int(used_seed)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == retries:
                break
            wait = attempt * 2
            if on_retry:
                on_retry(attempt, retries, f"{type(e).__name__}: {e} — retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Image generation failed after {retries} tries: {last_err}")


def save_image(img_bytes: bytes, prompt: str, out_dir: str | None = None) -> str:
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-") or "image"
    fname = f"flux-{slug}-{int(time.time())}.jpg"
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path


class ImageMakerAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "image_maker"
        self.model = model or MODEL_ID
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        prompt = build_prompt(instruction, context)
        width, height = parse_size(f"{context} {instruction}")
        steps = parse_steps(f"{context} {instruction}")
        img_bytes, used_seed = generate_image(
            prompt, width=width, height=height, seed=DEFAULT_SEED,
            steps=steps, on_retry=on_retry,
        )
        path = save_image(img_bytes, prompt)
        text = (
            f"Image generated with {self.model}\n"
            f"Prompt: {prompt}\n"
            f"Size: {width}x{height} | steps={steps} | seed={used_seed}\n"
            f"Saved to: {path}"
        )
        self._store(self.role, instruction, text)  # recall skipped: diffusion prompts stay clean
        return {"role": self.role, "model": self.model, "output": text, "file": path}
