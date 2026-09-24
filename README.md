# BOT-Agency
NIM-only agent agency. Boss: `nvidia/nemotron-3-ultra-550b-a55b`. One key in `.env`, 40 RPM shared.

## Setup
1. Put key in `.env`: `NVIDIA_API_KEY=...` (see `.env.example`, never commit `.env`)
2. `pip install -r requirements.txt`

## Run — AI Agency CLI (Claude Code style)
```
py main.py                                   # interactive REPL
py main.py "Write a limerick about GPUs" --direct
py main.py "Research GPUs vs CPUs, write benchmark code, then summarize"
py main.py --interactive                     # single-shot then REPL
```
REPL: `/help /agents /model /auto /direct /agency /stream /memory [clear] /clear /quit`

## Vector memory (ChromaDB)
- `agency/memory.py` — persistent `./chroma_db`, auto-recalled into router + direct asks.
- Boss stores every chat exchange and task result; recall top_k=3 (`MEMORY_TOP_K`).
- Offline hash embedding by default (no downloads). `MEMORY_EMBEDDING=default` for ONNX MiniLM.
- `py main.py --no-memory ...` or `MEMORY_ENABLED=0` to disable. `/memory` shows count, `/memory clear` wipes.

## Web-scraper agent
- `agency/scraper.py` — model `meta/muse-glimmer-30b` (`SCRAPER_MODEL`, override `WORKER_MODEL_SCRAPER`).
- Tool 1: ScrapeGraphAI `SmartScraperGraph` on NIM endpoint; Tool 2 fallback: Playwright chromium → page text → NIM summary.
- Router auto-picks `scraper` for URL tasks. `pip install scrapegraphai` + `py -m playwright install chromium` required for full path.

## YouTube-scraper bot
- `agency/yt_scraper.py` — model `poolside/laguna-xs-2.1` (`YT_SCRAPER_MODEL`, override `WORKER_MODEL_YT_SCRAPER`).
- Playwright fetch (channel/video/search, no new deps) → NIM summary. Router auto-picks `yt_scraper` for YouTube work.

## Lead-generation agent
- `agency/lead_gen.py` — model `google/gemma-4-31b-it` (`LEAD_MODEL`, override `WORKER_MODEL_LEAD_GEN`).
- Optional URL evidence via Playwright → one gemma call returns scored lead table. Never invents contacts. Router auto-picks `lead_gen` for leads/prospects.

## How it works
- `agency/main_agent.py` (boss) routes chat vs task, plans 1-4 JSON subtasks, then synthesizes.
- `agency/worker.py` + `agency/registry.py` (researcher/coder/writer/reviewer/scraper) run via same NIM endpoint.
- `agency/nim_client.py` = your base code (streaming + `reasoning_content` + `enable_thinking`).
- `agency/rate_limiter.py` = single 40 RPM limiter (1.5s gap, sequential workers).

## Test (no key needed)
`py test_agency.py`
