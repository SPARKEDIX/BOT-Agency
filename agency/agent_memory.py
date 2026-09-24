"""Shared Chroma memory for every worker agent.

Mixin: recall relevant past docs into the prompt, store each result.
Same `agency_memory` collection the boss uses — one memory, whole agency.
All failures are silent (memory never breaks a run).
"""
from __future__ import annotations

import config


class AgentMemoryMixin:
    _memory = None
    use_memory: bool = True

    def _mem(self):
        if self._memory is None and getattr(self, "use_memory", True) and config.MEMORY_ENABLED:
            try:
                from agency.memory import AgencyMemory

                self._memory = AgencyMemory()
            except Exception:
                self._memory = None
        return self._memory

    def _recall(self, query: str) -> str:
        try:
            m = self._mem()
            if m is not None and m.enabled:
                return m.recall_context(query)
        except Exception:
            pass
        return ""

    def _store(self, role: str, instruction: str, output: str) -> None:
        try:
            m = self._mem()
            if m is not None and m.enabled and (output or "").strip():
                m.add(f"[{role}] task: {(instruction or '')[:500]}\noutput: {output[:2000]}",
                      kind="worker", role=role)
        except Exception:
            pass
