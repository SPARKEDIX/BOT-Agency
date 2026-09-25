"""Central config. Single API key, 40 RPM shared budget, NIM endpoint."""
import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Main / boss model — also the default for every chat agent (fast + healthy).
MAIN_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# Web-scraper agent model (NIM id)
SCRAPER_MODEL = os.getenv("SCRAPER_MODEL", MAIN_MODEL)

# YouTube-scraper bot model (NIM id)
YT_SCRAPER_MODEL = os.getenv("YT_SCRAPER_MODEL", MAIN_MODEL)

# Coding agent model (NIM id)
CODING_MODEL = os.getenv("CODING_MODEL", MAIN_MODEL)

# Lead-generation agent model (NIM id)
LEAD_MODEL = os.getenv("LEAD_MODEL", MAIN_MODEL)

# Marketing agent model (NIM id)
MARKETING_MODEL = os.getenv("MARKETING_MODEL", MAIN_MODEL)

# URL-to-data scraper agent model (NIM id)
URL_DATA_MODEL = os.getenv("URL_DATA_MODEL", MAIN_MODEL)

# India stock-market analyst (trading) agent model (NIM id)
TRADING_MODEL = os.getenv("TRADING_MODEL", MAIN_MODEL)

# Flight-tracker agent model (NIM id)
FLIGHT_MODEL = os.getenv("FLIGHT_MODEL", MAIN_MODEL)

# Crypto-trading agent model (NIM id)
CRYPTO_MODEL = os.getenv("CRYPTO_MODEL", MAIN_MODEL)

# Real-time world-news agent model (NIM id)
NEWS_MODEL = os.getenv("NEWS_MODEL", MAIN_MODEL)

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
