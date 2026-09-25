"""India stock-market analyst agent: poolside/laguna-xs-2.1 + Playwright evidence.

Pipeline:
  1. Resolve symbols (RELIANCE, TCS.NS, NSE:INFY, ...) + explicit URLs from task.
  2. Build deep-check data URLs (Yahoo Finance, Screener.in) + fetch via Playwright.
  3. Deep data check: empty/blocked detection, price cross-check, conflict warnings.
  4. One NIM call turns evidence + upstream-agent context into structured analysis.
  Never invents prices: missing fields are marked 'unknown'. Not financial advice.

Connection with other agents (blackboard contract):
  - Upstream: accepts `context` from pipeline _handoff — typically researcher
    (market news), scraper/url_data (fetched pages). Treated as evidence,
    labelled "Upstream agent context".
  - Downstream: output is structured markdown (Price snapshot / Fundamentals /
    Technicals / News / Risks / View / Sources / Data quality) so writer can
    polish it, reviewer can validate it, and boss synthesizer can cite it.
"""
from __future__ import annotations

import re

from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 4
EVIDENCE_CHARS = 4000

# Nifty-50 / large-cap base set for symbol spotting. Generic detection also
# handles NSE:XXX / XXX.NS / XXX.BO patterns for any symbol.
KNOWN_SYMBOLS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC",
    "LT", "AXISBANK", "KOTAKBANK", "BHARTIARTL", "HCLTECH", "WIPRO",
    "MARUTI", "TATAMOTORS", "TATASTEEL", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "ADANIENT", "ADANIPORTS", "POWERGRID", "NTPC", "ONGC", "COALINDIA",
    "BAJFINANCE", "BAJAJFINSV", "ASIANPAINT", "NESTLEIND", "HEROMOTOCO",
    "EICHERMOT", "DRREDDY", "CIPLA", "DIVISLAB", "GRASIM", "JSWSTEEL",
    "HINDALCO", "BPCL", "BRITANNIA", "DABUR", "HAVELLS", "TECHM",
    "NIFTY", "SENSEX", "BANKNIFTY", "INDIAVIX",
}

_SYMBOL_PAT = re.compile(
    r"(?:NSE\s*:\s*|BSE\s*:\s*)?([A-Z]{2,12})(?:\.(?:NS|BO))?\b"
)
_PRICE_PAT = re.compile(r"(?:Rs\.?|INR|₹)\s*([\d,]+\.?\d*)")


def resolve_symbols(text: str) -> list[str]:
    """Extract likely NSE/BSE symbols from free text. Deduped, order kept."""
    out: list[str] = []
    for m in _SYMBOL_PAT.findall(text or ""):
        sym = m.strip().upper()
        if sym in KNOWN_SYMBOLS and sym not in out:
            out.append(sym)
    # Also catch explicit XXX.NS / XXX.BO / NSE:XXX forms for unknown names.
    for m in re.findall(r"\b([A-Z]{2,12})\.(?:NS|BO)\b|(?:NSE|BSE)\s*:\s*([A-Z]{2,12})\b", text or ""):
        sym = (m[0] or m[1]).upper()
        if sym and sym not in out:
            out.append(sym)
    return out[:3]


def build_data_urls(symbol: str) -> list[str]:
    """Deterministic deep-check quote pages for a symbol."""
    s = (symbol or "").upper().strip()
    if not s or s in ("NIFTY", "SENSEX", "BANKNIFTY", "INDIAVIX"):
        return ["https://finance.yahoo.com/quote/%5ENSEI"]
    base = re.sub(r"[^A-Z]", "", s)
    return [
        f"https://finance.yahoo.com/quote/{base}.NS",
        f"https://www.screener.in/company/{base}/",
    ]


def parse_prices(text: str) -> list[str]:
    """Pull Rs/INR/₹ price mentions for cross-checking. Returns raw strings."""
    return _PRICE_PAT.findall(text or "")[:10]


def deep_check(pages: list[dict]) -> list[str]:
    """Data-quality warnings across fetched pages. Pure function, testable.

    Each page: {"url": str, "text": str, "error": str|None}.
    Returns warning strings (empty = all good).
    """
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
    # Cross-check: distinct price mentions across sources.
    seen: set[str] = set()
    for p in ok_pages:
        for px in parse_prices(p.get("text", "")):
            seen.add(px.replace(",", ""))
    if len(seen) > 3:
        warnings.append(f"prices differ across sources ({len(seen)} distinct values) — report range + sources")
    return warnings


class TradingAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "trading"
        self.model = model or registry.model_for("trading")
        self.system = registry.AGENTS["trading"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        explicit_urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        symbols = resolve_symbols(text)

        # Auto-add deep-check URLs for top symbols if budget allows.
        urls: list[str] = list(explicit_urls)
        for sym in symbols[:2]:
            for u in build_data_urls(sym):
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

        user = f"Task:\n{instruction}"
        if symbols:
            user += f"\n\nSymbols detected: {', '.join(symbols)}"
        if context:
            # Upstream agents (researcher/scraper/url_data) land here via pipeline handoff.
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
            user += "\n\n(No symbols/URLs fetched — work from the task description, mark prices 'unknown', and ask for the stock symbol.)"
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
