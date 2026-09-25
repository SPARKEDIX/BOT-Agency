"""Offline frontend/backend checks: no NIM key needed (mocks agency calls)."""
import json
import os
import urllib.request

import backend.server as srv
from agency import registry

ROOT = os.path.dirname(os.path.abspath(__file__))


def test_tokens():
    css = open(os.path.join(ROOT, "frontend", "styles.css"), encoding="utf-8").read()
    for tok in ["--font-family-primary", "--font-size-base", "--font-size-xs", "--font-size-sm",
                "--font-size-md", "--font-size-lg", "--color-text-primary", "--color-text-secondary",
                "--color-text-tertiary", "--color-text-inverse", "--color-surface-base",
                "--color-surface-muted", "--color-surface-raised", "--color-surface-strong",
                "--color-border-default", "--color-focus-ring", "--space-1", "--space-5",
                "--space-8", "--radius-xs", "--radius-sm", "--radius-md", "--radius-lg",
                "--motion-instant", "--motion-fast", ":focus-visible", "prefers-reduced-motion"]:
        assert tok in css, f"missing token/rule {tok}"
    assert "#c3c2b7" in css and "#5598e7" in css
    print("tokens OK")


def test_html_a11y():
    html = open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
    for s in ['skip to', 'role="log"', 'aria-live="polite"', 'aria-label="Message composer"',
              'name="mode"', 'id="composer-input"',
              'id="agent-list"', 'id="trace-list"', "Shift+Enter", 'id="bg"', 'bg.js']:
        assert s.lower() in html.lower(), f"missing {s}"
    js = open(os.path.join(ROOT, "frontend", "app.js"), encoding="utf-8").read()
    assert "file:" in js and "AbortController" in js, "missing file:// guard / cancel"
    assert "/api/chat" in js and "/api/health" in js, "must use absolute /api paths"
    assert "fallback" in js.lower(), "missing easy-pipeline direct fallback"
    assert "showEmpty();\nrefreshMeta();" in js, "boot must not wipe messages"
    bg = open(os.path.join(ROOT, "frontend", "bg.js"), encoding="utf-8").read()
    assert "three" in bg.lower() and "prefers-reduced-motion" in bg
    print("html_a11y OK")


def test_api_mocked():
    # Mock boss + pipeline so no key/network is touched.
    import backend.server as s

    class FakeBoss:
        model = "test-boss"
        def memory_stats(self): return 7
        def memory_clear(self): pass
        def decide(self, goal, history=None, stream_output=False):
            return {"type": "task", "tasks": [{"id": 1, "agent": "trading", "instruction": "check RELIANCE"}]}
        def synthesize(self, goal, results, stream_output=False):
            return {"content": "FINAL"}
        def direct_ask(self, goal, history=None, stream_output=False):
            return {"content": "DIRECT"}
        def plan(self, goal, stream_output=False):
            return [{"id": 1, "agent": "researcher", "instruction": "x"}]
        def remember_exchange(self, *a): pass
        def remember_task(self, *a): pass

    s._BOSS = FakeBoss()
    import agency.worker as wmod
    Orig = wmod.WorkerAgent
    class FakeW:
        def __init__(self, role): self.role = role
        def run(self, instruction, context="", stream_output=False, on_retry=None):
            return {"role": self.role, "model": "m", "output": "out"}
    wmod.WorkerAgent = FakeW
    try:
        assert s.run_direct("hi", [])["final"] == "DIRECT"
        out = s.run_auto("Analyse RELIANCE", [])
        assert out["mode"] == "task" and out["final"] == "FINAL", out
        assert out["trace"][0].startswith("router:"), out
        assert len(registry.AGENTS) >= 12 and "trading" in registry.AGENTS
        print("api_mocked OK")
    finally:
        wmod.WorkerAgent = Orig
        s._BOSS = None


def test_design_md():
    md = open(os.path.join(ROOT, "design.md"), encoding="utf-8").read().lower()
    for s in ["context and goals", "design tokens", "component-level rules", "accessibility requirements",
              "content and tone", "anti-patterns", "qa checklist", "must", "should"]:
        assert s in md, f"design.md missing {s}"
    print("design_md OK")


if __name__ == "__main__":
    test_tokens()
    test_html_a11y()
    test_api_mocked()
    test_design_md()
    print("ALL FRONTEND OFFLINE TESTS PASSED")
