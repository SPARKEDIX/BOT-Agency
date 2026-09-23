"""Agent ecosystem. NIM models via one key + one endpoint.

Default worker model = MAIN_MODEL so you don't need extra entitlements.
Scraper defaults to meta/muse-glimmer-30b.
Override per-agent with e.g. WORKER_MODEL_CODER=nvidia/llama-3.1-nemotron-70b-instruct
"""
import os
import config

# role -> system prompt
AGENTS: dict[str, dict] = {
    "researcher": {
        "description": "Gathers facts, outlines, comparisons. No code execution.",
        "system": "You are a precise research agent. Return concise bullet facts, no fluff.",
    },
    "coder": {
        "description": "Writes and explains code. Returns code + brief usage.",
        "system": "You are an expert coding agent. Return clean code with minimal explanation.",
    },
    "writer": {
        "description": "Drafts docs, limericks, marketing copy, summaries.",
        "system": "You are a creative writing agent. Be clear and vivid.",
    },
    "reviewer": {
        "description": "Critiques and improves other agents' outputs.",
        "system": "You are a strict reviewer. Find flaws and fix them, then return improved version.",
    },
    "scraper": {
        "description": "Fetches website content via ScrapeGraphAI + Playwright, then summarizes.",
        "system": "You are a web-scraper agent. Extract the key facts from fetched page content. Return URL, title points, and concise summary.",
    },
}

# per-role model defaults (env override wins)
ROLE_DEFAULT_MODEL: dict[str, str] = {
    "scraper": config.SCRAPER_MODEL,
}


def model_for(role: str) -> str:
    """Model for a worker role. Env override > role default > MAIN_MODEL."""
    env_key = f"WORKER_MODEL_{role.upper()}"
    if os.getenv(env_key):
        return os.environ[env_key]
    return ROLE_DEFAULT_MODEL.get(role, config.MAIN_MODEL)
