# BOT-Agency — AI Agency

A team of AI agents that chat with you and do real tasks: research, code, scrape websites & YouTube, find leads, compare competitors, extract data from URLs, and generate images. One smart **boss** model decides whether to answer directly or split the work across specialist agents.

All language models run on the **NVIDIA NIM API** with a **single API key**. No OpenAI/Anthropic keys needed.

---

## 1. What you need

| Requirement | Details |
|---|---|
| Python | 3.10 or newer (`py --version`) |
| NVIDIA API key | Free at [build.nvidia.com](https://build.nvidia.com) → profile → API keys |
| OS | Windows, Linux, or macOS |
| Internet | For NIM API calls and web scraping |

---

## 2. Setup (5 minutes)

```bash
# 1. Get the code
git clone https://github.com/SPARKEDIX/BOT-Agency.git
cd BOT-Agency

# 2. (Recommended) isolated environment
py -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install everything
pip install -r requirements.txt

# 4. Install the browser used for scraping (one time)
py -m playwright install chromium

# 5. Add your key
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

Open `.env` and paste your key:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
```

> `.env` is git-ignored — your key never leaves your machine.

---

## 3. Run it

```bash
py backend/server.py                   # localhost frontend + API → http://127.0.0.1:8000
py main.py                                  # interactive chat (recommended)
py main.py "Summarize https://example.com"  # one task, then exit
py main.py "hi, what can you do?" --direct  # single boss call, no workers
py main.py "Research X, write code" --pipeline  # secure LangGraph run
```

Frontend (`frontend/` token-driven per `design.md`, WCAG 2.2 AA): chat + mode switch (auto/direct/agency/pipeline) + agent list + trace + memory, same-origin `/api/chat`. No new deps — stdlib server. Test offline with `py test_frontend.py`.

Just type normally. The boss **chats directly** for greetings and simple questions, and **splits real tasks** across agents automatically.

**Try these:**

```
❯ hi, who are you?
❯ Summarize https://www.youtube.com/@SPARKEDIX
❯ Research the SPARKEDIX GitHub profile and list top repos
❯ Find 5 AI automation agencies in india with contacts
❯ Compare Notion vs Obsidian pricing and features
❯ Extract the pricing table from https://example.com/pricing as JSON
❯ Generate an image of a cat in a library
```

### Modes

| Mode | Flag / command | What happens | NIM calls |
|---|---|---|---|
| `auto` (default) | — | Boss routes: chat = direct reply, task = split to agents | 1 for chat, up to ~6 for tasks |
| `direct` | `--direct` / `/direct` | Single boss call, no workers (best when NIM is busy) | 1 |
| `agency` | `--agency` / `/agency` | Forced plan → workers → synthesize | up to ~6 |
| `pipeline` | `--pipeline` / `/pipeline` | Secure LangGraph run with trace + safety guards | up to ~6 |

### Chat commands

Type `/` alone to list every command, use Tab to complete, and a typo like `/agnts` suggests the closest match.

| Command | Action |
|---|---|
| `/help` | Show all commands |
| `/agents` | List every agent + model |
| `/model <name>` | Show or switch the boss model |
| `/auto`, `/direct`, `/agency`, `/pipeline` | Switch mode |
| `/stream` | Toggle token streaming |
| `/memory`, `/memory clear` | Show Chroma docs / wipe memory |
| `/clear` | Clear screen |
| `/quit` | Exit |

---

## 4. The team

| Agent | Job | Model |
|---|---|---|
| Boss (`main_agent`) | Routes chat vs task, plans, synthesizes | `nvidia/nemotron-3-ultra-550b-a55b` |
| `researcher` | Facts, outlines, comparisons | boss model |
| `coder` / `coding` | Code, debugging (`coding` = Gemma) | boss / `google/gemma-4-31b-it` |
| `writer` | Docs, copy, summaries | boss model |
| `reviewer` | Critiques + improves drafts | boss model |
| `scraper` | Any URL → summary (ScrapeGraphAI, Playwright fallback) | `meta/muse-glimmer-30b` |
| `yt_scraper` | YouTube channel/video/search → stats + summary | `poolside/laguna-xs-2.1` |
| `lead_gen` | ICP → scored lead table (never invents contacts) | `google/gemma-4-31b-it` |
| `marketing` | Product URLs → brief + competitor table + angles | `meta/muse-glimmer-30b` |
| `url_data` | Any URL → tables / lists / JSON, verbatim | `meta/muse-glimmer-30b` |
| `trading` | India NSE/BSE analyst: price + fundamentals + technicals + risks (deep-check Yahoo/Screener) | `poolside/laguna-xs-2.1` |
| `flight_tracker` | Live flight status: times, delays, gates (FlightAware/FR24 + Playwright) | `meta/muse-glimmer-30b` |
| `image_maker` | Text → JPG image via FLUX.1-schnell (`./outputs`) | `black-forest-labs/flux.1-schnell` |

Every agent shares **one Chroma vector memory** (`./chroma_db`): each recalls relevant past work into its prompt and stores its result. One memory, whole agency. Disable with `--no-memory`.

Shared **40 requests/minute** budget: one rate limiter (1.5 s spacing), workers run sequentially, transient NIM errors auto-retry 5× with backoff.

---

## 5. Configuration (`.env` / environment)

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | — | **Required.** Single key for all models |
| `SCRAPER_MODEL` / `YT_SCRAPER_MODEL` / `LEAD_MODEL` / `MARKETING_MODEL` / `URL_DATA_MODEL` / `CODING_MODEL` / `TRADING_MODEL` / `FLIGHT_MODEL` | see `config.py` | Per-agent model override (or `WORKER_MODEL_<ROLE>`) |
| `CHROMA_PATH` | `./chroma_db` | Vector memory location |
| `MEMORY_TOP_K` / `MEMORY_ENABLED` | `3` / `1` | Recall depth / on-off |
| `MEMORY_EMBEDDING` | `hash` | `hash` = offline, `default` = ONNX MiniLM |

---

## 6. Project layout

```
main.py               # Claude-style CLI (REPL + single-shot)
config.py             # key, endpoint, models, budgets
test_agency.py        # offline tests, no key needed
agency/
  main_agent.py       # boss: router + planner + synthesizer
  worker.py           # dispatches every role
  registry.py         # agent list, prompts, model mapping
  nim_client.py       # shared NIM client (streaming + retries)
  rate_limiter.py     # 40 RPM shared limiter
  pipeline.py         # secure LangGraph graph (blackboard + guards)
  memory.py           # Chroma vector store
  agent_memory.py     # memory mixin for all workers
  scraper.py  yt_scraper.py  lead_gen.py  marketing.py  url_data.py
  trading.py  flight_tracker.py  image_maker.py
```

---

## 7. Test (no key needed)

```bash
py test_agency.py
```

Runs 21 offline checks with mocked API: routing, all agents, retries, memory, pipeline security/handoff/isolation.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `NVIDIA_API_KEY missing` | Put a real key in `.env` (not the `PASTE_` placeholder) |
| `Service temporarily overloaded` | Server-side, not your code — wait 30–60 s and retry, or use `--direct` (1 call instead of ~6) |
| `429 / rate limit` | 40 RPM budget hit — the limiter + retries handle it; just wait a minute |
| Playwright `Executable doesn't exist` | Run `py -m playwright install chromium` |
| `scrapegraphai` install fails | Optional — the scraper auto-falls-back to Playwright |
| Empty page / login wall | Some sites block bots; the agent reports it instead of guessing |
| Slow image generation | FLUX can take minutes when busy; check `./outputs` for the finished file |

## 9. Security notes

The `--pipeline` mode adds: input sanitization, API-key redaction from prompts/logs/memory, SSRF URL filtering (blocks localhost, private networks, cloud metadata IPs, credentials in URLs), per-agent error isolation, a 10-step budget, and fail-closed errors with a full trace.
