"""Real-time world news agent: Google News RSS (fast, no browser) + Playwright fallback.

Pipeline:
  1. Topic from the task (or explicit URLs).
  2. Live headlines via Google News RSS with `requests` (no NIM/RPM cost).
  3. Explicit URLs (if any) fetched via Playwright for depth.
  4. One NIM call turns headlines + pages into a structured briefing.
Never invents events: only fetched headlines are reported, each with source.

Shared helper: `fetch_rss()` is imported by the trading agents
(`trading`, `crypto`) so they get live market news inside their own
analysis with zero extra NIM calls.
"""
from __future__ import annotations

import html as _html
import re

from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_many_playwright, MAX_CHARS

MAX_URLS = 3
MAX_HEADLINES = 8
RSS_TIMEOUT = 15

RSS_URL = "https://news.google.com/rss/search"

_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.S)
_TAG_RE = {
    "title": re.compile(r"<title>(.*?)</title>", re.S),
    "link": re.compile(r"<link>(.*?)</link>", re.S),
    "pubDate": re.compile(r"<pubDate>(.*?)</pubDate>", re.S),
    "source": re.compile(r"<source[^>]*>(.*?)</source>", re.S),
    "description": re.compile(r"<description>(.*?)</description>", re.S),
}


def _tag(block: str, name: str) -> str:
    m = _TAG_RE[name].search(block)
    if not m:
        return ""
    return _html.unescape(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(1), flags=re.S)).strip()


def build_rss_url(query: str, region: str = "IN") -> str:
    from urllib.parse import urlencode

    hl = "en-IN" if region == "IN" else "en-US"
    ceid = f"{region}:en"
    return f"{RSS_URL}?{urlencode({'q': query, 'hl': hl, 'gl': region, 'ceid': ceid})}"


def fetch_rss(query: str, max_items: int = 5, region: str = "IN") -> list[dict]:
    """Live headlines for a query. Pure fetch, no NIM. Returns
    [{title, link, source, time, summary}]. Empty list on any failure."""
    import requests

    try:
        resp = requests.get(
            build_rss_url(query, region),
            headers={"User-Agent": "Mozilla/5.0 (BOT-Agency news agent)"},
            timeout=RSS_TIMEOUT,
        )
        if resp.status_code != 200 or not resp.text:
            return []
        items: list[dict] = []
        for block in _ITEM_RE.findall(resp.text)[:max_items]:
            title = _tag(block, "title")
            if not title:
                continue
            items.append({
                "title": title[:300],
                "link": _tag(block, "link")[:500],
                "source": _tag(block, "source")[:80] or "unknown",
                "time": _tag(block, "pubDate")[:60] or "unknown",
                "summary": re.sub(r"<[^>]+>", "", _tag(block, "description"))[:300],
            })
        return items
    except Exception:
        return []


def resolve_topic(instruction: str) -> str:
    """RSS query from free text: strip boilerplate, cap length."""
    text = re.sub(r"(?i)\b(get|fetch|find|show|give|tell)\s+(me\s+)?(the\s+)?(latest|top|breaking|real[\s-]?time)?\s*(news|headlines?)\s*(about|on|for|of)?\b", "",
                  instruction or "").strip(" :,-")
    return text[:150] or (instruction or "").strip()[:150]


class NewsAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "news"
        self.model = model or registry.model_for("news")
        self.system = registry.AGENTS["news"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        topic = resolve_topic(instruction)
        urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]

        headlines = fetch_rss(topic or instruction.strip(), max_items=MAX_HEADLINES)

        pages: list[str] = []
        texts = fetch_many_playwright(urls)  # one browser launch for all URLs
        for url in urls:
            raw = (texts.get(url) or "")[:4000]
            pages.append(f"Source {url}:\n{raw}" if raw.strip() else f"Source {url}: empty page (login wall or block)")

        user = f"Task:\n{instruction}\n\nTopic: {topic or 'general world news'}"
        if context:
            user += f"\n\nUpstream agent context (use as evidence):\n{context[:4000]}"
        if headlines:
            user += "\n\nLive headlines (newest first):\n" + "\n".join(
                f"- {h['title']} ({h['source']}, {h['time']})" for h in headlines)
        else:
            user += "\n\n(No live headlines fetched — RSS unavailable. Say so and work from fetched pages/memory.)"
        if pages:
            user += "\n\nFetched pages:\n" + "\n---\n".join(pages)
        snippet = self._recall(instruction)
        if snippet:
            user += f"\n\nRelevant past memory:\n{snippet}"

        res = nim_client.chat(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user[:MAX_CHARS]},
            ],
            model=self.model,
            temperature=0.3,
            top_p=0.95,
            max_tokens=4096,
            enable_thinking=False,
            stream_output=stream_output,
            on_retry=on_retry,
        )
        self._store(self.role, instruction, res["content"])
        return {"role": self.role, "model": self.model, "output": res["content"]}
