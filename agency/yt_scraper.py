"""YouTube-scraper bot: poolside/laguna-xs-2.1 + Playwright.

Fast path, no new deps (Playwright chromium already installed):
  1. Resolve target: direct YouTube URL, @handle, or search term.
  2. Fetch page text via Playwright (channel / video / search-results page).
  3. Summarize via NIM yt model (rate-limited, counts to 40 RPM).
"""
from __future__ import annotations

import re
import urllib.parse

import config
from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 3
HANDLE_RE = re.compile(r"@[\w.-]+", re.I)


def resolve_targets(instruction: str, context: str = "") -> list[str]:
    """YouTube URLs first, then @handles, then a search URL from the task text."""
    text = ((context or "") + "\n" + (instruction or "")).strip()
    urls = [u for u in URL_RE.findall(text) if "youtube.com" in u or "youtu.be" in u]
    seen = [u.rstrip(".,);]") for u in urls if u.rstrip(".,);]") not in []]
    targets: list[str] = []
    for u in seen:
        if u not in targets:
            targets.append(u)
    for h in HANDLE_RE.findall(text):
        url = f"https://www.youtube.com/{h}"
        if url not in targets:
            targets.append(url)
    if not targets and text.strip():
        q = urllib.parse.quote_plus(" ".join(text.split()[:12]))
        targets.append(f"https://www.youtube.com/results?search_query={q}")
    return targets[:MAX_URLS]


class YTScraperAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "yt_scraper"
        self.model = model or registry.model_for("yt_scraper")
        self.system = registry.AGENTS["yt_scraper"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        targets = resolve_targets(instruction, context)
        snippet = self._recall(instruction)
        mem_ctx = f"\n\nRelevant past memory:\n{snippet}" if snippet else ""
        parts: list[str] = []
        for url in targets:
            try:
                raw = fetch_playwright(url)
                source = "playwright"
            except Exception as e:  # noqa: BLE001
                parts.append(f"URL: {url}\nERROR: could not fetch ({e})")
                continue
            if not raw.strip():
                parts.append(f"URL: {url}\nERROR: empty page (login wall or block)")
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
