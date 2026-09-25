"""Flight-tracker agent: meta/muse-glimmer-30b + Playwright evidence.

Pipeline:
  1. Resolve flight numbers (AI202, 6E345, UA48) + airport codes + explicit URLs.
  2. Build live-tracking URLs (FlightAware, Flightradar24) + fetch via Playwright.
  3. Deep data check: empty/blocked detection, no usable-data warnings.
  4. One NIM call turns evidence + upstream-agent context into structured status.
  Never invents times/gates: missing fields are marked 'unknown'.

Connection with other agents (blackboard contract):
  - Upstream: accepts `context` from pipeline _handoff — typically researcher
    (weather/strike news) or scraper/url_data (fetched pages). Treated as
    evidence, labelled "Upstream agent context".
  - Downstream: output is structured markdown (Flight / Route / Status /
    Times / Gate-Terminal / Delay / Sources / Data quality) so writer can
    polish it, reviewer can validate it, and boss synthesizer can cite it.
"""
from __future__ import annotations

import re

from agency import nim_client, registry
from agency.agent_memory import AgentMemoryMixin
from agency.scraper import URL_RE, fetch_playwright, MAX_CHARS

MAX_URLS = 4
EVIDENCE_CHARS = 4000

# IATA airline codes (India-first + major internationals). Generic XX1234
# matches are only accepted for these, to avoid false positives like "A1".
KNOWN_AIRLINES = {
    "AI", "6E", "SG", "UK", "QP", "I5", "G8", "IX", "LH", "BA",
    "EK", "QR", "EY", "SQ", "MH", "TG", "CX", "UA", "DL", "AA",
    "AF", "KL", "TK", "SV", "WY", "KU", "GF", "UL",
}

# IATA airport codes (India + major internationals + hubs).
KNOWN_AIRPORTS = {
    "DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "AMD", "PNQ", "GOI",
    "COK", "JAI", "LKO", "PAT", "BBI", "GAY", "IXC", "TRV", "IXM",
    "DXB", "AUH", "DOH", "SIN", "LHR", "JFK", "EWR", "ORD", "SFO",
    "FRA", "CDG", "AMS", "HKG", "BKK", "KUL", "CMB", "KTM", "DAC",
}

_FLIGHT_PAT = re.compile(r"\b([A-Z0-9]{2})\s?-?(\d{1,4}[A-Z]?)\b")
_ROUTE_PAT = re.compile(r"\b([A-Z]{3})\s*(?:→|->|–|—|-|\bto\b)\s*([A-Z]{3})\b", re.I)


def resolve_flights(text: str) -> list[str]:
    """Extract likely flight numbers (AI202, 6E345). Deduped, order kept."""
    out: list[str] = []
    for airline, num in _FLIGHT_PAT.findall(text or ""):
        code = f"{airline.upper()}{num}"
        if airline.upper() in KNOWN_AIRLINES and code not in out:
            out.append(code)
    return out[:3]


def resolve_route(text: str) -> str | None:
    """Extract an airport-pair route (DEL→BOM). Returns 'AAA-BBB' or None."""
    m = _ROUTE_PAT.search(text or "")
    if not m:
        return None
    a, b = m.group(1).upper(), m.group(2).upper()
    if a in KNOWN_AIRPORTS and b in KNOWN_AIRPORTS:
        return f"{a}-{b}"
    return None


def build_tracking_urls(flight: str) -> list[str]:
    """Deterministic live-tracking pages for a flight number."""
    code = re.sub(r"[^A-Z0-9]", "", (flight or "").upper())
    if not code:
        return []
    return [
        f"https://www.flightaware.com/live/flight/{code}",
        f"https://www.flightradar24.com/data/flights/{code.lower()}",
    ]


def deep_check(pages: list[dict]) -> list[str]:
    """Data-quality warnings across fetched pages. Pure function, testable.

    Each page: {"url": str, "text": str, "error": str|None}.
    Returns warning strings (empty = all good).
    """
    warnings: list[str] = []
    ok_pages = [p for p in pages if not p.get("error") and (p.get("text") or "").strip()]
    if not pages:
        return ["no tracking pages fetched — status is low-confidence"]
    for p in pages:
        if p.get("error"):
            warnings.append(f"{p['url']}: {p['error']}")
        elif not (p.get("text") or "").strip():
            warnings.append(f"{p['url']}: empty page (login wall or block)")
    if not ok_pages:
        warnings.append("no usable tracking data — mark times/gates 'unknown'")
    return warnings


class FlightTrackerAgent(AgentMemoryMixin):
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.role = "flight_tracker"
        self.model = model or registry.model_for("flight_tracker")
        self.system = registry.AGENTS["flight_tracker"]["system"]
        self._memory = memory
        self.use_memory = use_memory

    def run(self, instruction: str, context: str = "", stream_output: bool = False, on_retry=None) -> dict:
        text = ((context or "") + "\n" + (instruction or "")).strip()
        explicit_urls = [u.rstrip(".,);]") for u in URL_RE.findall(text)][:MAX_URLS]
        flights = resolve_flights(text)
        route = resolve_route(text)

        # Auto-add live-tracking URLs for top flights if budget allows.
        urls: list[str] = list(explicit_urls)
        for fl in flights[:2]:
            for u in build_tracking_urls(fl):
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
        if flights:
            user += f"\n\nFlights detected: {', '.join(flights)}"
        if route:
            user += f"\nRoute detected: {route}"
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
            user += "\n\nFetched tracking data:\n" + "\n---\n".join(chunks)
        else:
            user += "\n\n(No flight numbers/URLs fetched — work from the task description, mark times/gates 'unknown', and ask for the flight number.)"
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
