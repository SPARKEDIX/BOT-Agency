"""Marketing agent: meta/muse-glimmer-30b + web evidence.

Pipeline:
  1. Fetch product/competitor URLs from the task via scraper helpers (Playwright).
  2. One NIM call turns product info + pages into a product brief,
     competitor table, and positioning angles. No invented facts.
"""
from __future__ import annotations

from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 4


class MarketingAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "marketing"
        self.model = model or registry.model_for("marketing")
        self.system = registry.AGENTS["marketing"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        evidence: list[str] = []
        for url in urls:
            try:
                evidence.append(f"Page {url}:\n{fetch_playwright(url)[:4000]}")
            except Exception as e:  # noqa: BLE001
                evidence.append(f"Page {url}: fetch failed ({e})")
        user = f"Task:\n{instruction}"
        if context:
            user += f"\n\nContext:\n{context}"
        if evidence:
            user += "\n\nFetched pages:\n" + "\n---\n".join(evidence)
        else:
            user += "\n\n(No product URLs fetched — work from the task description and mark unfound facts 'unknown'.)"
        snippet = self._recall(instruction)
        if snippet:
            user += f"\n\nRelevant past memory:\n{snippet}"
        res = nim_client.chat(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user[:MAX_CHARS]},
            ],
            model=self.model,
            temperature=0.5,
            top_p=0.95,
            max_tokens=4096,
            enable_thinking=False,
            stream_output=stream_output,
            on_retry=on_retry,
        )
        self._store(self.role, instruction, res["content"])
        return {"role": self.role, "model": self.model, "output": res["content"]}
