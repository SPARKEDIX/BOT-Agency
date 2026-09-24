"""Web-scraper agent: meta/muse-glimmer-30b + ScrapeGraphAI + Playwright.

Pipeline per URL (max 3 per task):
  1. Try ScrapeGraphAI SmartScraperGraph with NIM (OpenAI-compatible endpoint).
  2. Fallback: Playwright (chromium headless) -> page text.
  3. Summarize fetched text via NIM scraper model (rate-limited, counts to 40 RPM).
"""
from __future__ import annotations

import re

import config
from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
MAX_URLS = 3
MAX_CHARS = 12000


def extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for m in URL_RE.findall(text or ""):
        url = m.rstrip(".,);]")
        if url not in seen:
            seen.append(url)
    return seen[:MAX_URLS]


def fetch_playwright(url: str, timeout_ms: int = 30000) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_timeout(1500)
            except Exception:
                pass
            text = page.inner_text("body") or ""
        finally:
            browser.close()
    return text[:MAX_CHARS]


def smart_scrape(url: str, prompt: str) -> str | None:
    """ScrapeGraphAI with NIM backend. Returns text or None if unavailable."""
    try:
        from scrapegraphai.graphs import SmartScraperGraph
    except Exception:
        return None
    try:
        graph = SmartScraperGraph(
            prompt=prompt or "Extract the main content, facts, and key points.",
            source=url,
            config={
                "llm": {
                    "model": config.SCRAPER_MODEL,
                    "api_key": config.require_key(),
                    "openai_api_base": config.BASE_URL,
                },
                "verbose": False,
            },
        )
        return str(graph.run() or "")[:MAX_CHARS] or None
    except Exception:
        return None


class ScraperAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "scraper"
        self.model = model or registry.model_for("scraper")
        self.system = registry.AGENTS["scraper"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        urls = extract_urls((context or "") + "\n" + (instruction or ""))
        snippet = self._recall(instruction)
        mem_ctx = f"\n\nRelevant past memory:\n{snippet}" if snippet else ""
        if not urls:
            res = nim_client.chat(
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": f"No URL found. Ask for the URL, then briefly explain what you will extract.\nTask:\n{instruction}"},
                ],
                model=self.model,
                temperature=0.3,
                top_p=0.95,
                max_tokens=1024,
                enable_thinking=False,
                stream_output=stream_output,
                on_retry=on_retry,
            )
            self._store(self.role, instruction, res["content"])
            return {"role": self.role, "model": self.model, "output": res["content"]}

        parts: list[str] = []
        for url in urls:
            raw = smart_scrape(url, instruction)  # tool 1: scrapegraphai
            source = "scrapegraphai"
            if not raw:
                try:
                    raw = fetch_playwright(url)  # tool 2: playwright fallback
                    source = "playwright"
                except Exception as e:  # noqa: BLE001
                    parts.append(f"URL: {url}\nERROR: could not fetch ({e})")
                    continue
            summary = nim_client.chat(
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": f"URL: {url}\nFetched via {source}:\n{raw[:MAX_CHARS]}\n\nTask:\n{instruction}{mem_ctx}"},
                ],
                model=self.model,
                temperature=0.3,
                top_p=0.95,
                max_tokens=4096,
                enable_thinking=False,
                stream_output=stream_output,
                on_retry=on_retry,
            )
            parts.append(f"URL: {url}\n{summary['content']}")
        final = "\n\n---\n\n".join(parts)
        self._store(self.role, instruction, final)
        return {"role": self.role, "model": self.model, "output": final}
