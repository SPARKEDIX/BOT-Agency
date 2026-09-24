"""URL-to-data scraper agent: meta/muse-glimmer-30b + Playwright.

Pipeline per URL (max 3 per task):
  1. Fetch page text via Playwright (chromium headless).
  2. One NIM call extracts the requested fields as structured
     data (tables / lists / JSON). No invented values.
"""
from __future__ import annotations

from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 3


class UrlDataAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "url_data"
        self.model = model or registry.model_for("url_data")
        self.system = registry.AGENTS["url_data"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        snippet = self._recall(instruction)
        mem_ctx = f"\n\nRelevant past memory:\n{snippet}" if snippet else ""
        if not urls:
            res = nim_client.chat(
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": f"No URL found. Ask for the URL and which fields to extract.\nTask:\n{instruction}"},
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
            try:
                raw = fetch_playwright(url)
            except Exception as e:  # noqa: BLE001
                parts.append(f"URL: {url}\nERROR: could not fetch ({e})")
                continue
            if not raw.strip():
                parts.append(f"URL: {url}\nERROR: empty page (login wall or block)")
                continue
            data = nim_client.chat(
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": f"URL: {url}\nPage text:\n{raw[:MAX_CHARS]}\n\nExtract:\n{instruction}{mem_ctx}"},
                ],
                model=self.model,
                temperature=0.2,
                top_p=0.95,
                max_tokens=4096,
                enable_thinking=False,
                stream_output=stream_output,
                on_retry=on_retry,
            )
            parts.append(f"URL: {url}\n{data['content']}")
        final = "\n\n---\n\n".join(parts)
        self._store(self.role, instruction, final)
        return {"role": self.role, "model": self.model, "output": final}
