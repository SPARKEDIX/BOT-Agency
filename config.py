"""Central config. Single API key, 40 RPM shared budget, NIM endpoint."""
import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Main / boss model
MAIN_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# Web-scraper agent model (NIM id)
SCRAPER_MODEL = os.getenv("SCRAPER_MODEL", "meta/muse-glimmer-30b")

# Coding agent model (NIM id)
CODING_MODEL = os.getenv("CODING_MODEL", "google/gemma-4-31b-it")

# Image generation (NVIDIA Cloud, not integrate endpoint)
IMAGE_MODEL = "black-forest-labs/flux.1-schnell"
IMAGE_URL = os.getenv(
    "IMAGE_URL",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
)
IMAGE_OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "./outputs")
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "4"))

# Shared throughput budget: 40 requests / minute for the whole agency.
# All agents (main + workers) go through one RateLimiter.
RPM = 40

# Defaults from your base code
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 16384

# Chroma vector memory for the main agent
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "2000"))
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "1") == "1"


def require_key() -> str:
    if not NVIDIA_API_KEY or NVIDIA_API_KEY.startswith("PASTE_"):
        raise RuntimeError(
            "NVIDIA_API_KEY missing. Put your key in .env as NVIDIA_API_KEY=..."
        )
    return NVIDIA_API_KEY
