"""Agent ecosystem. NIM models via one key + one endpoint.

Default worker model = MAIN_MODEL so you don't need extra entitlements.
Scraper defaults to meta/muse-glimmer-30b, YT scraper to poolside/laguna-xs-2.1.
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
    "coding": {
        "description": "Dedicated coding agent (Gemma) for codegen, debugging, refactoring.",
        "system": "You are an expert coding agent. Write clean, tested code with minimal explanation. Return file paths, code blocks, and run instructions.",
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
    "yt_scraper": {
        "description": "Scrapes YouTube channels/videos/search via Playwright, then summarizes.",
        "system": "You are a YouTube-scraper bot. Extract channel stats, video titles, views, dates, and key points from fetched YouTube page text. Return URL, stats, and concise bullet summary.",
    },
    "lead_gen": {
        "description": "Finds and qualifies sales leads from an ICP; returns scored lead lists.",
        "system": "You are a lead-generation agent. From the target ICP and any fetched context, return a scored lead list as a markdown table: Name | Title | Company | Contact hint | Source | Score (1-10) | Why. Only real, verifiable entities — never invent emails or phone numbers; mark unknowns as 'unknown'. End with 3 suggested next actions.",
    },
    "image_maker": {
        "description": "Generates images via FLUX.1-schnell, saves JPG to ./outputs.",
        "system": "You are an image-generation agent. Turn the task into a vivid image prompt and report the saved file path.",
    },
}

# per-role model defaults (env override wins)
ROLE_DEFAULT_MODEL: dict[str, str] = {
    "scraper": config.SCRAPER_MODEL,
    "yt_scraper": config.YT_SCRAPER_MODEL,
    "lead_gen": config.LEAD_MODEL,
    "coding": config.CODING_MODEL,
    "image_maker": config.IMAGE_MODEL,
}


def model_for(role: str) -> str:
    """Model for a worker role. Env override > role default > MAIN_MODEL."""
    env_key = f"WORKER_MODEL_{role.upper()}"
    if os.getenv(env_key):
        return os.environ[env_key]
    return ROLE_DEFAULT_MODEL.get(role, config.MAIN_MODEL)
