"""Lead-generation agent: google/gemma-4-31b-it + optional web evidence.

Pipeline:
  1. If the task contains URLs, fetch them via scraper helpers (Playwright).
  2. One NIM call (gemma) turns ICP + evidence into a scored lead table.
No invented contacts: unknowns are marked 'unknown'.
"""
from __future__ import annotations

import config
from agency import nim_client, registry
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 3


class LeadGenAgent:
    def __init__(self, model: str | None = None):
        self.role = "lead_gen"
        self.model = model or registry.model_for("lead_gen")
        self.system = registry.AGENTS["lead_gen"]["system"]

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        evidence: list[str] = []
        for url in urls:
            try:
                evidence.append(f"Source {url}:\n{fetch_playwright(url)[:4000]}")
            except Exception as e:  # noqa: BLE001
                evidence.append(f"Source {url}: fetch failed ({e})")
        user = f"Task:\n{instruction}"
        if context:
            user += f"\n\nContext:\n{context}"
        if evidence:
            user += "\n\nWeb evidence:\n" + "\n---\n".join(evidence)
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
        return {"role": self.role, "model": self.model, "output": res["content"]}
