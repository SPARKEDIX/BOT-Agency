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
    assert set(registry.AGENTS) == {"researcher", "coder", "coding", "writer", "reviewer", "scraper", "yt_scraper", "lead_gen", "marketing", "url_data", "trading", "flight_tracker", "crypto", "news", "image_maker"}
    assert registry.model_for("coder").startswith("nvidia/")
    assert registry.model_for("scraper") == config.SCRAPER_MODEL == "meta/muse-glimmer-30b"
    assert registry.model_for("url_data") == config.URL_DATA_MODEL == "meta/muse-glimmer-30b"
    assert registry.model_for("marketing") == config.MARKETING_MODEL == "meta/muse-glimmer-30b"
    assert registry.model_for("lead_gen") == config.LEAD_MODEL == "google/gemma-4-31b-it"
    assert registry.model_for("yt_scraper") == config.YT_SCRAPER_MODEL == "poolside/laguna-xs-2.1"
    assert registry.model_for("coding") == config.CODING_MODEL == "google/gemma-4-31b-it"
    assert registry.model_for("trading") == config.TRADING_MODEL == "poolside/laguna-xs-2.1"
    assert registry.model_for("flight_tracker") == config.FLIGHT_MODEL == "meta/muse-glimmer-30b"
    assert registry.model_for("crypto") == config.CRYPTO_MODEL == "poolside/laguna-xs-2.1"
    assert registry.model_for("news") == config.NEWS_MODEL == "meta/muse-glimmer-30b"
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

        out = WorkerAgent("scraper", use_memory=False).run("Summarize https://example.com", stream_output=False)
        assert out["role"] == "scraper", out
        assert out["model"] == "meta/muse-glimmer-30b", out
        assert "https://example.com" in out["output"], out
        print("scraper OK")
    finally:
        nc.chat, sc.smart_scrape, sc.fetch_playwright = orig_chat, orig_smart, orig_fetch


def test_yt_scraper_agent_mocked():
    """YT scraper: target resolution + fetch + NIM summary (all mocked, no network)."""
    import agency.nim_client as nc
    import agency.yt_scraper as yt

    assert yt.resolve_targets("Check https://www.youtube.com/@SPARKEDIX stats") == [
        "https://www.youtube.com/@SPARKEDIX"]
    assert yt.resolve_targets("Find @SomeChannel videos")[0] == "https://www.youtube.com/@SomeChannel"
    assert yt.resolve_targets("funny cats")[0].startswith("https://www.youtube.com/results?search_query=")

    orig_chat, orig_fetch = nc.chat, yt.fetch_playwright
    nc.chat = lambda messages, model, **kw: {"content": "yt-summary", "reasoning": ""}
    yt.fetch_playwright = lambda url, timeout_ms=30000: "Sparkedix 815 subscribers 208 videos"
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("yt_scraper", use_memory=False).run("Summarize https://www.youtube.com/@SPARKEDIX", stream_output=False)
        assert out["role"] == "yt_scraper", out
        assert out["model"] == "poolside/laguna-xs-2.1", out
        assert "yt-summary" in out["output"], out
        print("yt_scraper OK")
    finally:
        nc.chat, yt.fetch_playwright = orig_chat, orig_fetch


def test_lead_gen_agent_mocked():
    """Lead gen: scored table via gemma (mocked, no network)."""
    import agency.nim_client as nc
    import agency.lead_gen as lg

    orig_chat, orig_fetch = nc.chat, lg.fetch_playwright
    seen = {}
    nc.chat = lambda messages, model, **kw: (seen.update(model=model), {"content": "| A | CEO | Acme | unknown | web | 8 | fits |", "reasoning": ""})[1]
    lg.fetch_playwright = lambda url, timeout_ms=30000: (_ for _ in ()).throw(AssertionError("no fetch expected"))
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("lead_gen", use_memory=False).run("Find AI automation agencies in india", stream_output=False)
        assert out["role"] == "lead_gen", out
        assert out["model"] == "google/gemma-4-31b-it", out
        assert seen["model"] == "google/gemma-4-31b-it", seen
        assert "Acme" in out["output"], out
        print("lead_gen OK")
    finally:
        nc.chat, lg.fetch_playwright = orig_chat, orig_fetch


def test_marketing_agent_mocked():
    """Marketing: brief + competitor table via muse-glimmer (mocked, no network)."""
    import agency.nim_client as nc
    import agency.marketing as mk

    orig_chat, orig_fetch = nc.chat, mk.fetch_playwright
    seen = {}
    nc.chat = lambda messages, model, **kw: (seen.update(model=model), {"content": "Brief: Acme CRM. | RivalCo | x | y |", "reasoning": ""})[1]
    mk.fetch_playwright = lambda url, timeout_ms=30000: (_ for _ in ()).throw(AssertionError("no fetch expected"))
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("marketing", use_memory=False).run("Research Acme CRM and competitors", stream_output=False)
        assert out["role"] == "marketing", out
        assert out["model"] == "meta/muse-glimmer-30b", out
        assert seen["model"] == "meta/muse-glimmer-30b", seen
        assert "Acme" in out["output"], out
        print("marketing OK")
    finally:
        nc.chat, mk.fetch_playwright = orig_chat, orig_fetch


def test_url_data_agent_mocked():
    """URL-to-data: fetch + structured extract (mocked, no network)."""
    import agency.nim_client as nc
    import agency.url_data as ud

    orig_chat, orig_fetch = nc.chat, ud.fetch_playwright
    seen = {}
    nc.chat = lambda messages, model, **kw: (seen.update(model=model), {"content": "| Price | $9 |", "reasoning": ""})[1]
    ud.fetch_playwright = lambda url, timeout_ms=30000: "Pricing page: Basic $9"
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("url_data", use_memory=False).run("Extract pricing from https://example.com/pricing", stream_output=False)
        assert out["role"] == "url_data", out
        assert out["model"] == "meta/muse-glimmer-30b", out
        assert seen["model"] == "meta/muse-glimmer-30b", seen
        assert "$9" in out["output"], out
        print("url_data OK")
    finally:
        nc.chat, ud.fetch_playwright = orig_chat, orig_fetch


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


def test_image_maker_mocked():
    """Image maker: prompt/size parsing + FLUX POST + base64 save (all mocked, no network)."""
    import base64
    import tempfile
    import agency.image_maker as im

    assert im.parse_size("make it landscape 1344x768") == (1344, 768)
    assert im.parse_size("portrait poster") == (768, 1344)
    assert im.build_prompt("Generate an image of a red robot", "") == "a red robot"

    fake_bytes = b"FAKEJPG"
    b64 = base64.b64encode(fake_bytes).decode()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"artifacts": [{"base64": b64, "seed": 123}]}

    import requests
    orig_post, orig_key, orig_wait = requests.post, None, im.limiter.wait
    try:
        import config as cfg
        orig_key = cfg.NVIDIA_API_KEY
        cfg.NVIDIA_API_KEY = "test-key"
        requests.post = lambda url, headers=None, json=None, timeout=120: (
            _ for _ in ()).throw(AssertionError("unset")) if False else FakeResp()
        # capture payload
        seen = {}

        def fake_post(url, headers=None, json=None, timeout=120):
            seen["url"] = url
            seen["payload"] = json
            assert url == im.INVOKE_URL, url
            assert json["prompt"], json
            return FakeResp()

        requests.post = fake_post
        im.limiter.wait = lambda: 0
        with tempfile.TemporaryDirectory() as td:
            out = im.ImageMakerAgent(use_memory=False).run("Generate an image of a red robot, landscape")
            # rerun save path check via returned file? run saves to ./outputs; check output text
            assert out["role"] == "image_maker", out
            assert "Saved to:" in out["output"], out
            assert seen["payload"]["width"] == 1344, seen["payload"]
            # direct save test to tmp
            p = im.save_image(fake_bytes, "red robot", out_dir=td)
            with open(p, "rb") as f:
                assert f.read() == fake_bytes
            # cleanup default outputs file created by run()
            try:
                import os
                os.remove(out["file"])
            except Exception:
                pass
        print("image_maker OK")
    finally:
        requests.post = orig_post
        im.limiter.wait = orig_wait
        import config as cfg
        cfg.NVIDIA_API_KEY = orig_key


def test_trading_agent_mocked():
    """Trading: symbol resolve + deep-check URLs + NIM analysis (mocked, no network)."""
    import agency.nim_client as nc
    import agency.trading as tr

    assert tr.resolve_symbols("Analyse RELIANCE and NSE:INFY please") == ["RELIANCE", "INFY"]
    assert tr.resolve_symbols("TCS.NS price?") == ["TCS"]
    assert tr.build_data_urls("RELIANCE")[0] == "https://finance.yahoo.com/quote/RELIANCE.NS"
    assert "screener.in" in tr.build_data_urls("TCS")[1]
    assert tr.deep_check([]) == ["no market pages fetched — analysis is low-confidence"]
    assert tr.deep_check([{"url": "u", "text": "", "error": "fetch failed (x)"}]) != []

    orig_chat, orig_fetch = nc.chat, tr.fetch_playwright
    seen = {}
    nc.chat = lambda messages, model, **kw: (seen.update(model=model, prompt=messages[1]["content"]), {"content": "Price snapshot: Rs 3000 | Not financial advice.", "reasoning": ""})[1]
    tr.fetch_playwright = lambda url, timeout_ms=30000: f"RELIANCE price Rs 3000 on NSE from {url}"
    orig_news = tr.live_news
    tr.live_news = lambda q, max_items=3: [{"title": "RELIANCE Q3 profit up", "source": "TestWire", "time": "today", "link": "", "summary": ""}]
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("trading", use_memory=False).run("Analyse RELIANCE on NSE", stream_output=False)
        assert out["role"] == "trading", out
        assert out["model"] == "poolside/laguna-xs-2.1", out
        assert seen["model"] == "poolside/laguna-xs-2.1", seen
        assert "Rs 3000" in out["output"] or "Not financial advice" in out["output"], out
        # Real-time link: live news must reach the prompt.
        assert "RELIANCE Q3 profit up" in seen["prompt"], seen["prompt"][:300]
        # Upstream-agent connection: context must reach the prompt.
        out2 = WorkerAgent("trading", use_memory=False).run(
            "Give view", context="Prior researcher found: strong Q3 results", stream_output=False)
        assert "researcher" in seen["prompt"] or "Q3" in seen["prompt"], seen["prompt"][:300]
        print("trading OK")
    finally:
        nc.chat, tr.fetch_playwright = orig_chat, orig_fetch
        tr.live_news = orig_news


def test_flight_tracker_agent_mocked():
    """Flight tracker: flight resolve + tracking URLs + NIM status (mocked, no network)."""
    import agency.nim_client as nc
    import agency.flight_tracker as ft

    assert ft.resolve_flights("Track AI202 please") == ["AI202"]
    assert ft.resolve_flights("Status of 6E 345?") == ["6E345"]
    assert ft.resolve_flights("hello world") == []
    assert ft.resolve_route("DEL to BOM") == "DEL-BOM"
    assert ft.resolve_route("no route here") is None
    assert ft.build_tracking_urls("AI202")[0] == "https://www.flightaware.com/live/flight/AI202"
    assert ft.deep_check([]) == ["no tracking pages fetched — status is low-confidence"]
    assert ft.deep_check([{"url": "u", "text": "", "error": "fetch failed (x)"}]) != []

    orig_chat, orig_fetch = nc.chat, ft.fetch_playwright
    seen = {}
    nc.chat = lambda messages, model, **kw: (seen.update(model=model, prompt=messages[1]["content"]), {"content": "Status: en route, dep 10:00. Verify with the airline before travel.", "reasoning": ""})[1]
    ft.fetch_playwright = lambda url, timeout_ms=30000: f"AI202 en route, departed 10:00 from {url}"
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("flight_tracker", use_memory=False).run("Track AI202 DEL to BOM", stream_output=False)
        assert out["role"] == "flight_tracker", out
        assert out["model"] == "meta/muse-glimmer-30b", out
        assert seen["model"] == "meta/muse-glimmer-30b", seen
        assert "en route" in out["output"], out
        # Upstream-agent connection: context must reach the prompt.
        out2 = WorkerAgent("flight_tracker", use_memory=False).run(
            "Give status", context="Prior researcher found: fog at DEL", stream_output=False)
        assert "fog" in seen["prompt"], seen["prompt"][:300]
        print("flight_tracker OK")
    finally:
        nc.chat, ft.fetch_playwright = orig_chat, orig_fetch


def test_crypto_agent_mocked():
    """Crypto: coin resolve + tracking URLs + NIM analysis (mocked, no network)."""
    import agency.nim_client as nc
    import agency.crypto as cc

    assert cc.resolve_coins("Analyse BTC and ETH please") == ["BTC", "ETH"]
    assert cc.resolve_coins("BTCUSDT price?") == ["BTC"]
    assert cc.resolve_coins("hello world") == []
    assert cc.build_crypto_urls("BTC")[0] == "https://finance.yahoo.com/quote/BTC-USD"
    assert "coingecko" in cc.build_crypto_urls("ETH")[1]
    assert cc.deep_check([]) == ["no market pages fetched — analysis is low-confidence"]
    assert cc.deep_check([{"url": "u", "text": "", "error": "fetch failed (x)"}]) != []

    orig_chat, orig_fetch = nc.chat, cc.fetch_playwright
    seen = {}
    nc.chat = lambda messages, model, **kw: (seen.update(model=model, prompt=messages[1]["content"]), {"content": "BTC $97000 | Strategy: hold. Not financial advice.", "reasoning": ""})[1]
    cc.fetch_playwright = lambda url, timeout_ms=30000: f"BTC price $97000 from {url}"
    orig_news = cc.live_news
    cc.live_news = lambda q, max_items=3: [{"title": "ETF inflows hit record", "source": "TestWire", "time": "today", "link": "", "summary": ""}]
    try:
        from agency.worker import WorkerAgent

        out = WorkerAgent("crypto", use_memory=False).run("Should I buy BTC?", stream_output=False)
        assert out["role"] == "crypto", out
        assert out["model"] == "poolside/laguna-xs-2.1", out
        assert seen["model"] == "poolside/laguna-xs-2.1", seen
        assert "Not financial advice" in out["output"], out
        # Real-time link: live news must reach the prompt.
        assert "ETF inflows" in seen["prompt"], seen["prompt"][:300]
        print("crypto OK")
    finally:
        nc.chat, cc.fetch_playwright = orig_chat, orig_fetch
        cc.live_news = orig_news


def test_news_agent_mocked():
    """News: topic resolve + RSS parse + NIM briefing (mocked, no network)."""
    import agency.nim_client as nc
    import agency.news as nw

    assert nw.resolve_topic("Get me the latest news about AI") == "AI"
    assert "RELIANCE" in nw.resolve_topic("RELIANCE stock news")
    assert nw.build_rss_url("bitcoin").startswith("https://news.google.com/rss/search?q=")

    import requests
    orig_chat, orig_get = nc.chat, requests.get
    seen = {}

    class FakeResp:
        status_code = 200
        text = ('<rss><channel><item><title>Markets rally</title><link>http://x</link>'
                '<pubDate>today</pubDate><source>TestWire</source>'
                '<description>Stocks up.</description></item></channel></rss>')

    nc.chat = lambda messages, model, **kw: (seen.update(model=model, prompt=messages[1]["content"]), {"content": "Top story: Markets rally (TestWire).", "reasoning": ""})[1]
    requests.get = lambda url, headers=None, timeout=15: FakeResp()
    try:
        heads = nw.fetch_rss("markets")
        assert heads and heads[0]["title"] == "Markets rally" and heads[0]["source"] == "TestWire", heads
        from agency.worker import WorkerAgent

        out = WorkerAgent("news", use_memory=False).run("Top world news", stream_output=False)
        assert out["role"] == "news", out
        assert out["model"] == "meta/muse-glimmer-30b", out
        assert "Markets rally" in seen["prompt"], seen["prompt"][:300]
        assert "Top story" in out["output"], out
        print("news OK")
    finally:
        nc.chat, requests.get = orig_chat, orig_get


def test_pipeline_security():
    """SSRF guard, secret redaction, unsafe-task filter."""
    from agency.pipeline import is_safe_url, redact_secrets, sanitize_text, split_tasks

    assert is_safe_url("https://example.com/page")
    for bad in ["file:///etc/passwd", "http://localhost:8000/x", "http://127.0.0.1/",
                "http://169.254.169.254/latest", "http://10.0.0.5/", "http://192.168.1.1/",
                "ftp://example.com/f", "https://user:pass@example.com/"]:
        assert not is_safe_url(bad), bad
    assert "[REDACTED]" in redact_secrets("key=nvapi-abc123XYZ456 here")
    assert "[REDACTED]" in redact_secrets("Authorization: Bearer sometoken123")
    assert len(sanitize_text("x" * 9000)) == 4000
    ok, blocked = split_tasks([{"id": 1, "agent": "scraper", "instruction": "fetch https://example.com"},
                               {"id": 2, "agent": "scraper", "instruction": "fetch http://169.254.169.254/"}])
    assert len(ok) == 1 and len(blocked) == 1, (ok, blocked)
    print("pipeline_security OK")


def test_pipeline_chat():
    """Chat goal: 1 router call, no workers, traced."""
    import agency.nim_client as nc
    from agency.pipeline import run_pipeline

    orig = nc.chat
    nc.chat = lambda messages, model, **kw: {"content": '{"type":"chat","reply":"hi there"}', "reasoning": ""}
    try:
        out = run_pipeline("hello", boss=MainAgent(use_memory=False))
        assert out["mode"] == "chat" and out["final"] == "hi there", out
        assert any("router" in t for t in out["trace"]) and any("chat_answer" in t for t in out["trace"])
        assert out["results"] == []
        print("pipeline_chat OK")
    finally:
        nc.chat = orig


def _fake_worker_factory(seen, fail_on=()):
    import agency.worker as wmod

    class FakeWorker:
        def __init__(self, role):
            self.role = role

        def run(self, instruction, context="", stream_output=False, on_retry=None):
            seen.append({"role": self.role, "instruction": instruction, "context": context})
            if self.role in fail_on:
                raise RuntimeError("boom")
            return {"role": self.role, "model": "fake", "output": f"out-{self.role}"}

    return wmod, FakeWorker


def test_pipeline_task_handoff():
    """Task goal: workers run in order, each sees prior outputs (blackboard)."""
    import agency.nim_client as nc
    from agency.pipeline import run_pipeline

    def fake_chat(messages, model, **kw):
        if "BOSS router" in messages[0]["content"]:
            return {"content": '{"type":"task","tasks":[{"id":1,"agent":"researcher","instruction":"find X"},{"id":2,"agent":"writer","instruction":"write X"}]}', "reasoning": ""}
        return {"content": "FINAL", "reasoning": ""}

    seen: list[dict] = []
    wmod, FakeWorker = _fake_worker_factory(seen)
    orig_chat, orig_worker = nc.chat, wmod.WorkerAgent
    nc.chat, wmod.WorkerAgent = fake_chat, FakeWorker
    try:
        out = run_pipeline("do X", boss=MainAgent(use_memory=False))
        assert out["mode"] == "task" and out["final"] == "FINAL", out
        assert [r["role"] for r in out["results"]] == ["researcher", "writer"], out["results"]
        assert "out-researcher" in seen[1]["context"], seen[1]  # handoff proof
        assert any("worker" in t for t in out["trace"]) and any("synth" in t for t in out["trace"])
        print("pipeline_handoff OK")
    finally:
        nc.chat, wmod.WorkerAgent = orig_chat, orig_worker


def test_pipeline_error_isolation():
    """Failing worker is isolated: rest run, synthesis still happens."""
    import agency.nim_client as nc
    from agency.pipeline import run_pipeline

    def fake_chat(messages, model, **kw):
        if "BOSS router" in messages[0]["content"]:
            return {"content": '{"type":"task","tasks":[{"id":1,"agent":"researcher","instruction":"find X"},{"id":2,"agent":"writer","instruction":"write X"}]}', "reasoning": ""}
        return {"content": "FINAL", "reasoning": ""}

    seen: list[dict] = []
    wmod, FakeWorker = _fake_worker_factory(seen, fail_on=("researcher",))
    orig_chat, orig_worker = nc.chat, wmod.WorkerAgent
    nc.chat, wmod.WorkerAgent = fake_chat, FakeWorker
    try:
        out = run_pipeline("do X", boss=MainAgent(use_memory=False))
        assert out["final"] == "FINAL", out
        assert any("researcher" in e for e in out["errors"]), out["errors"]
        assert any("FAILED isolated" in t for t in out["trace"]), out["trace"]
        print("pipeline_isolation OK")
    finally:
        nc.chat, wmod.WorkerAgent = orig_chat, orig_worker


def test_all_agents_memory():
    """Every agent recalls past memory into prompts and stores its output."""
    import agency.nim_client as nc
    from agency.worker import WorkerAgent

    class FakeMem:
        enabled = True

        def __init__(self):
            self.stored: list[dict] = []

        def recall_context(self, q):
            return "past aurora notes"

        def add(self, text, kind="worker", **meta):
            self.stored.append({"text": text, "kind": kind, **meta})
            return "id1"

    seen: dict = {}
    orig = nc.chat
    nc.chat = lambda messages, model, **kw: (seen.update(prompt=messages[1]["content"]), {"content": "done", "reasoning": ""})[1]
    try:
        mem = FakeMem()
        out = WorkerAgent("writer", memory=mem).run("write about aurora")
        assert out["output"] == "done"
        assert "past aurora notes" in seen["prompt"], seen["prompt"][:200]
        assert len(mem.stored) == 1 and mem.stored[0]["role"] == "writer", mem.stored

        from agency.lead_gen import LeadGenAgent

        mem2 = FakeMem()
        out = LeadGenAgent(memory=mem2).run("Find leads")
        assert out["output"] == "done"
        assert "past aurora notes" in seen["prompt"], seen["prompt"][:200]
        assert len(mem2.stored) == 1 and mem2.stored[0]["role"] == "lead_gen", mem2.stored
        print("all_agents_memory OK")
    finally:
        nc.chat = orig


def test_slash_suggest():
    """`/` recommends every command; partial/typo input suggests matches."""
    from main import SLASH_COMMANDS, suggest_commands

    assert set(suggest_commands("/")) == set(SLASH_COMMANDS) and len(SLASH_COMMANDS) >= 10
    assert "/agents" in suggest_commands("/agen")
    assert "/quit" in suggest_commands("/quitt")  # fuzzy match
    assert suggest_commands("/zzz-nope") == []
    print("slash_suggest OK")


if __name__ == "__main__":
    test_extract_json()
    test_registry()
    test_rate_limiter_spacing()
    test_agency_flow_mocked()
    test_smart_chat_path()
    test_parse_router()
    test_scraper_agent_mocked()
    test_yt_scraper_agent_mocked()
    test_lead_gen_agent_mocked()
    test_marketing_agent_mocked()
    test_url_data_agent_mocked()
    test_trading_agent_mocked()
    test_flight_tracker_agent_mocked()
    test_crypto_agent_mocked()
    test_news_agent_mocked()
    test_pipeline_security()
    test_pipeline_chat()
    test_pipeline_task_handoff()
    test_pipeline_error_isolation()
    test_slash_suggest()
    test_all_agents_memory()
    test_retry_on_overload()
    test_memory_roundtrip()
    test_router_uses_memory()
    test_image_maker_mocked()
    print("ALL OFFLINE TESTS PASSED")
