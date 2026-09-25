"""Lead-generation agent: Nemotron 3 Ultra + optional web evidence.

Pipeline:
  1. If the task contains URLs, fetch them via scraper helpers (Playwright).
  2. One NIM call (gemma) turns ICP + evidence into a scored lead table.
No invented contacts: unknowns are marked 'unknown'.
"""
from __future__ import annotations

import config
from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_many_playwright, MAX_CHARS

MAX_URLS = 3


class LeadGenAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "lead_gen"
        self.model = model or registry.model_for("lead_gen")
        self.system = registry.AGENTS["lead_gen"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        evidence: list[str] = []
        texts = fetch_many_playwright(urls)  # one browser launch for all URLs
        for url in urls:
            raw = texts.get(url) or ""
            if raw.strip():
                evidence.append(f"Source {url}:\n{raw[:4000]}")
            else:
                evidence.append(f"Source {url}: fetch failed or empty page")
        user = f"Task:\n{instruction}"
        if context:
            user += f"\n\nContext:\n{context}"
        if evidence:
            user += "\n\nWeb evidence:\n" + "\n---\n".join(evidence)
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
