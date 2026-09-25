"""Crypto-trading agent: CoinGecko/Yahoo/CoinDesk via Playwright + live news.

Pipeline:
  1. Resolve coins (BTC, ETH, SOL, BTCUSDT, BTC/USD ...) + explicit URLs.
  2. Build tracking URLs (Yahoo Finance, CoinGecko, CoinDesk) + fetch via Playwright.
  3. Deep data check: empty/blocked detection, price cross-check warnings.
  4. Live crypto news via news.fetch_rss (zero NIM cost) for real-time context.
  5. One NIM call turns evidence + news into analysis + strategy suggestion.
  Never invents prices: missing fields are marked 'unknown'.

Connection with other agents (blackboard contract):
  - Upstream: `context` from pipeline _handoff (news/researcher/scraper).
    Plus auto-fetched live news for the top coin.
  - Downstream: structured markdown (Price / Market / On-chain+Sentiment /
    News / Risks / Strategy / Sources / Data quality) for writer/reviewer/boss.
"""
from __future__ import annotations

import re

from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 4
EVIDENCE_CHARS = 4000

# Major coins: symbol -> (CoinGecko slug, CoinDesk slug).
KNOWN_COINS = {
    "BTC": ("bitcoin", "bitcoin"),
    "ETH": ("ethereum", "ethereum"),
    "SOL": ("solana", "solana"),
    "XRP": ("xrp", "xrp"),
    "DOGE": ("dogecoin", "dogecoin"),
    "ADA": ("cardano", "cardano"),
    "AVAX": ("avalanche", "avalanche"),
    "LINK": ("chainlink", "chainlink"),
    "MATIC": ("matic-network", "polygon"),
    "DOT": ("polkadot", "polkadot"),
    "LTC": ("litecoin", "litecoin"),
    "BNB": ("binancecoin", "binancecoin"),
    "TRX": ("tron", "tron"),
    "ATOM": ("cosmos", "cosmos"),
    "NEAR": ("near", "near"),
    "ARB": ("arbitrum", "arbitrum"),
    "OP": ("optimism", "optimism"),
}

_COIN_PAT = re.compile(r"\b([A-Z]{2,6})(?:[/\-\s]?(?:USD₮|USDT|USD|INR|BTC|ETH))?\b")
_PAIR_PAT = re.compile(r"\b([A-Z]{2,6})(?:[/\-]?(?:USDT|USD|INR))\b", re.I)
_PRICE_PAT = re.compile(r"(?:\$|USD|INR|₹)\s*([\d,]+\.?\d*)")


def resolve_coins(text: str) -> list[str]:
    """Extract likely coin symbols. Deduped, order kept."""
    out: list[str] = []
    for m in _PAIR_PAT.findall(text or ""):
        sym = m.upper()
        if sym in KNOWN_COINS and sym not in out:
            out.append(sym)
    for m in _COIN_PAT.findall(text or ""):
        sym = m.upper()
        if sym in KNOWN_COINS and sym not in out:
            out.append(sym)
    return out[:3]


def build_crypto_urls(symbol: str) -> list[str]:
    """Deterministic tracking pages for a coin."""
    s = (symbol or "").upper().strip()
    urls = [f"https://finance.yahoo.com/quote/{s}-USD"]
    if s in KNOWN_COINS:
        gecko, desk = KNOWN_COINS[s]
        urls.append(f"https://www.coingecko.com/en/coins/{gecko}")
        urls.append(f"https://www.coindesk.com/price/{desk}")
    return urls


def parse_prices(text: str) -> list[str]:
    return _PRICE_PAT.findall(text or "")[:10]


def deep_check(pages: list[dict]) -> list[str]:
    """Data-quality warnings across fetched pages. Pure function, testable."""
    warnings: list[str] = []
    ok_pages = [p for p in pages if not p.get("error") and (p.get("text") or "").strip()]
    if not pages:
        return ["no market pages fetched — analysis is low-confidence"]
    for p in pages:
        if p.get("error"):
            warnings.append(f"{p['url']}: {p['error']}")
        elif not (p.get("text") or "").strip():
            warnings.append(f"{p['url']}: empty page (login wall or block)")
    if not ok_pages:
        warnings.append("no usable market data — mark prices 'unknown'")
        return warnings
    seen: set[str] = set()
    for p in ok_pages:
        for px in parse_prices(p.get("text", "")):
            seen.add(px.replace(",", ""))
    if len(seen) > 3:
        warnings.append(f"prices differ across sources ({len(seen)} distinct values) — report range + sources")
    return warnings


def live_news(query: str, max_items: int = 3) -> list[dict]:
    """Live headlines for a coin/market. Never raises; [] on failure."""
    try:
        from agency.news import fetch_rss

        return fetch_rss(query, max_items=max_items, region="US") or []
    except Exception:
        return []


class CryptoAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "crypto"
        self.model = model or registry.model_for("crypto")
        self.system = registry.AGENTS["crypto"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        explicit_urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        coins = resolve_coins(text)

        urls: list[str] = list(explicit_urls)
        for coin in coins[:2]:
            for u in build_crypto_urls(coin):
                if len(urls) >= MAX_URLS:
                    break
                if u not in urls:
                    urls.append(u)
            if len(urls) >= MAX_URLS:
                break

        pages: list[dict] = []
        for url in urls:
            try:
                raw = fetch_playwright(url)[:EVIDENCE_CHARS]
                if not raw.strip():
                    pages.append({"url": url, "text": "", "error": "empty page (login wall or block)"})
                else:
                    pages.append({"url": url, "text": raw, "error": None})
            except Exception as e:  # noqa: BLE001
                pages.append({"url": url, "text": "", "error": f"fetch failed ({e})"})

        warnings = deep_check(pages)
        # Real-time link: live news for the top coin, zero NIM cost.
        news = live_news(f"{coins[0]} crypto" if coins else (instruction.strip()[:80] or "crypto market"))

        user = f"Task:\n{instruction}"
        if coins:
            user += f"\n\nCoins detected: {', '.join(coins)}"
        if context:
            user += f"\n\nUpstream agent context (use as evidence, don't repeat verbatim):\n{context[:4000]}"
        if pages:
            chunks = []
            for p in pages:
                if p["error"]:
                    chunks.append(f"Source {p['url']}:\n{p['error']}")
                else:
                    chunks.append(f"Source {p['url']}:\n{p['text']}")
            user += "\n\nFetched market data:\n" + "\n---\n".join(chunks)
        else:
            user += "\n\n(No coins/URLs fetched — work from the task description, mark prices 'unknown', and ask for the coin symbol.)"
        if news:
            user += "\n\nLive crypto news (use for strategy):\n" + "\n".join(
                f"- {n['title']} ({n['source']}, {n['time']})" for n in news)
        if warnings:
            user += "\n\nData-quality warnings (surface these in output):\n- " + "\n- ".join(warnings)
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
