"""Offline tests: no API key needed (mocks nim_client.chat)."""
import config
from agency.main_agent import _extract_json, _parse_router, MainAgent
from agency.rate_limiter import RateLimiter
from agency import registry


def test_extract_json():
    txt = '```json\n[{"id":1,"agent":"coder","instruction":"write fizzbuzz"}]\n```'
    tasks = _extract_json(txt)
    assert tasks[0]["agent"] == "coder", tasks
    print("extract_json OK:", tasks)


def test_registry():
    assert set(registry.AGENTS) == {"researcher", "coder", "writer", "reviewer", "scraper"}
    assert registry.model_for("coder").startswith("nvidia/")
    assert registry.model_for("scraper") == config.SCRAPER_MODEL == "meta/muse-glimmer-30b"
    print("registry OK:", list(registry.AGENTS))


def test_rate_limiter_spacing():
    lim = RateLimiter(rpm=40)
    assert abs(lim.min_interval - 1.5) < 1e-6, lim.min_interval
    print("rate_limiter OK: min_interval=1.5s for 40 RPM")


def test_agency_flow_mocked():
    import agency.nim_client as nc
    calls = []

    def fake_chat(messages, model, **kw):
        calls.append(messages[0]["content"][:20])
        # router call returns task JSON, others return text
        if "BOSS router" in messages[0]["content"]:
            return {"content": '{"type":"task","tasks":[{"id":1,"agent":"researcher","instruction":"fact A"},{"id":2,"agent":"writer","instruction":"write it"}]}', "reasoning": ""}
        if "BOSS synthesizer" in messages[0]["content"]:
            return {"content": "FINAL", "reasoning": ""}
        return {"content": f"worker-out:{messages[-1]['content'][:20]}", "reasoning": ""}

    orig = nc.chat
    nc.chat = fake_chat
    try:
        out = MainAgent(use_memory=False).run("demo goal", stream_output=False, log=lambda *a: None)
        assert out["mode"] == "task", out
        assert out["final"] == "FINAL"
        assert len(out["worker_results"]) == 2
        assert len(calls) == 4, calls  # router + 2 workers + synth
        print("agency_flow OK:", [r["role"] for r in out["worker_results"]])
    finally:
        nc.chat = orig


def test_smart_chat_path():
    """Normal chat must answer directly: 1 call, no workers."""
    import agency.nim_client as nc
    calls = []

    def fake_chat(messages, model, **kw):
        calls.append(1)
        assert "BOSS router" in messages[0]["content"]
        return {"content": '{"type":"chat","reply":"Hey! How can I help?"}', "reasoning": ""}

    orig = nc.chat
    nc.chat = fake_chat
    try:
        out = MainAgent(use_memory=False).run("hi", stream_output=False, log=lambda *a: None)
        assert out["mode"] == "chat", out
        assert out["final"] == "Hey! How can I help?"
        assert out["worker_results"] == []
        assert len(calls) == 1, calls
        print("smart_chat OK")
    finally:
        nc.chat = orig


def test_parse_router():
    d = _parse_router('{"type":"chat","reply":"hello"}')
    assert d == {"type": "chat", "reply": "hello"}, d
    d = _parse_router('```json\n{"type":"task","tasks":[{"id":1,"agent":"coder","instruction":"write fizzbuzz"}]}\n```')
    assert d["type"] == "task" and d["tasks"][0]["agent"] == "coder", d
    d = _parse_router('{"type":"task","tasks":[{"id":1,"agent":"scraper","instruction":"scrape https://example.com"}]}')
    assert d["type"] == "task" and d["tasks"][0]["agent"] == "scraper", d
    d = _parse_router("just some plain text")
    assert d["type"] == "chat" and "plain" in d["reply"], d
    print("parse_router OK")


def test_scraper_agent_mocked():
    """Scraper: URL extraction + playwright fallback + NIM summary (all mocked, no network)."""
    import agency.nim_client as nc
    import agency.scraper as sc

    urls = sc.extract_urls("fetch https://example.com and http://test.org/a, then summarize")
    assert urls == ["https://example.com", "http://test.org/a"], urls

    orig_chat, orig_smart, orig_fetch = nc.chat, sc.smart_scrape, sc.fetch_playwright
    nc.chat = lambda messages, model, **kw: {"content": f"summary:{messages[1]['content'][:30]}", "reasoning": ""}
    sc.smart_scrape = lambda url, prompt: None  # force playwright path
    sc.fetch_playwright = lambda url, timeout_ms=30000: f"page text for {url}"
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("scraper").run("Summarize https://example.com", stream_output=False)
        assert out["role"] == "scraper", out
        assert out["model"] == "meta/muse-glimmer-30b", out
        assert "https://example.com" in out["output"], out
        print("scraper OK")
    finally:
        nc.chat, sc.smart_scrape, sc.fetch_playwright = orig_chat, orig_smart, orig_fetch


def test_retry_on_overload():
    """nim_client.chat must retry 'Service temporarily overloaded' instead of raising."""
    import agency.nim_client as nc

    class FakeChunk:
        def __init__(self, text):
            delta = type("D", (), {"content": text, "reasoning_content": None})()
            self.choices = [type("C", (), {"delta": delta})()]

    class FakeStream:
        def __init__(self, text):
            self._chunks = [FakeChunk(text)]

        def __iter__(self):
            return iter(self._chunks)

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise Exception("Service temporarily overloaded")
            return FakeStream("recovered")

    class FakeClient:
        def __init__(self):
            self.chat = type("X", (), {"completions": FakeCompletions()})()

    orig_client, orig_wait = nc.get_client, nc.limiter.wait
    fake = FakeClient()
    nc.get_client = lambda: fake
    nc.limiter.wait = lambda: 0
    try:
        res = nc.chat([{"role": "user", "content": "hi"}], model="m",
                      stream_output=False, retries=3, on_retry=lambda *a: None)
        assert res["content"] == "recovered", res
        assert fake.chat.completions.calls == 2
        print("retry_on_overload OK")
    finally:
        nc.get_client, nc.limiter.wait = orig_client, orig_wait


def test_memory_roundtrip():
    """Chroma memory: store + recall + count + clear (in-memory, no files)."""
    from agency.memory import AgencyMemory, HashEmbedding

    assert len(HashEmbedding()(["hello world"])[0]) == 384
    mem = AgencyMemory(path=":memory:", top_k=2)
    assert mem.count() == 0
    mem.remember_exchange("my gpu project aurora", "noted aurora")
    mem.remember_task("build aurora benchmark", [{"id": 1, "agent": "coder", "instruction": "write aurora benchmark"}], "done")
    assert mem.count() == 2, mem.count()
    hits = mem.recall("aurora benchmark")
    assert any("aurora" in h.lower() for h in hits), hits
    assert "aurora" in mem.recall_context("aurora").lower()
    mem.clear()
    assert mem.count() == 0
    print("memory_roundtrip OK")


def test_router_uses_memory():
    """decide() must inject recalled memory into the router prompt."""
    import agency.nim_client as nc
    seen = {}

    def fake_chat(messages, model, **kw):
        seen["system"] = messages[0]["content"]
        return {"content": '{"type":"chat","reply":"ok"}', "reasoning": ""}

    class FakeMem:
        enabled = True

        def recall_context(self, q):
            return "past project aurora notes"

        def remember_exchange(self, *a):
            pass

    orig = nc.chat
    nc.chat = fake_chat
    try:
        boss = MainAgent(memory=FakeMem())
        out = boss.decide("continue aurora")
        assert out["type"] == "chat"
        assert "aurora notes" in seen["system"], seen["system"][:200]
        print("router_uses_memory OK")
    finally:
        nc.chat = orig


if __name__ == "__main__":
    test_extract_json()
    test_registry()
    test_rate_limiter_spacing()
    test_agency_flow_mocked()
    test_smart_chat_path()
    test_parse_router()
    test_scraper_agent_mocked()
    test_retry_on_overload()
    test_memory_roundtrip()
    test_router_uses_memory()
    print("ALL OFFLINE TESTS PASSED")
