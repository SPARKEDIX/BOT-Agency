"""Secure LangGraph pipeline: every agent talks through one shared blackboard.

Graph: intake -> router -> {chat_answer | worker <-> dispatch} -> synthesizer -> END
  - chat goals answer in 1 NIM call, task goals fan out to workers sequentially
    (40 RPM safe), each worker sees prior workers' outputs (handoff), then the
    boss synthesizes. Conversation threads persist via MemorySaver.

Security (fail-closed, senior defaults):
  - sanitize_text: length caps + control-char strip on all model-bound inputs.
  - redact_secrets: API keys / bearer tokens never reach prompts, logs, or memory.
  - is_safe_url + task URL pre-filter: blocks SSRF (localhost, private nets,
    cloud metadata IP, credentials in URL, non-http schemes). Blocked tasks are
    skipped with a recorded error, never fetched.
  - STEP_BUDGET: hard cap on node visits; breach drains the queue to synthesis.
  - Per-worker error isolation: one agent failing never kills the run.
  - All worker outputs truncated before re-entering prompts (context-bomb guard).
"""
from __future__ import annotations

import ipaddress
import operator
import re
import urllib.parse
from typing import Annotated, Any, Callable, Literal

import config
from agency.main_agent import MainAgent

# -- budgets ---------------------------------------------------------------
MAX_GOAL_CHARS = 4000
MAX_HISTORY_ITEM = 2000
MAX_CONTEXT_CHARS = 6000
MAX_RESULT_CHARS = 4000
STEP_BUDGET = 10
MAX_TASKS = 4

SECRET_RES = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}", re.I),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.~+/=]{8,}", re.I),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)(['\"]?)[A-Za-z0-9_\-\.~+/=]{12,}\2"),
]

UNSAFE_HOSTS = {"localhost", "metadata.google.internal"}
METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


def redact_secrets(text: str) -> str:
    out = text or ""
    for rx in SECRET_RES:
        out = rx.sub("[REDACTED]", out)
    return out


def sanitize_text(text: str, limit: int = MAX_GOAL_CHARS) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "")
    t = redact_secrets(t).strip()
    return t[:limit]


def is_safe_url(url: str) -> bool:
    """SSRF guard: only public http(s), no credentials, no private targets."""
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if p.username or p.password or "@" in (p.netloc or ""):
        return False
    host = (p.hostname or "").lower().strip(".")
    if not host or host in UNSAFE_HOSTS:
        return False
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_global and str(ip) not in METADATA_IPS
    except ValueError:
        pass  # hostname, not literal IP
    if host in ("localhost",) or host.endswith((".local", ".internal", ".lan")):
        return False
    return True


def split_tasks(text: str) -> tuple[list[dict], list[str]]:
    """Validate router tasks: cap count, drop unsafe-URL tasks with a reason."""
    from agency.scraper import URL_RE

    ok: list[dict] = []
    blocked: list[str] = []
    for t in (text if isinstance(text, list) else [])[:MAX_TASKS]:
        urls = [u.rstrip(".,);]") for u in URL_RE.findall(t.get("instruction", ""))]
        bad = [u for u in urls if not is_safe_url(u)]
        if bad:
            blocked.append(f"task {t.get('id')} blocked unsafe URL: {bad[0][:80]}")
            continue
        ok.append(t)
    return ok, blocked


# -- graph state ------------------------------------------------------------
class PipeState(dict):
    goal: str
    history: list[dict]
    route: str
    chat_reply: str
    tasks: list[dict]
    remaining: list[dict]
    results: Annotated[list[dict], operator.add]
    trace: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    steps: int
    final: str


_BOSS: MainAgent | None = None


def _boss(use_memory: bool = True) -> MainAgent:
    global _BOSS
    if _BOSS is None or _BOSS.use_memory != use_memory:
        _BOSS = MainAgent(use_memory=use_memory)
    return _BOSS


def _handoff(goal: str, results: list[dict]) -> str:
    parts = [f"Overall goal: {goal}", "Prior agents' outputs (build on these, don't repeat):"]
    for r in results[-3:]:
        parts.append(f"[{r.get('role')}#{r.get('id')}]: {str(r.get('output', ''))[:1500]}")
    return sanitize_text("\n".join(parts), MAX_CONTEXT_CHARS)


# -- nodes -------------------------------------------------------------------
def n_intake(state: PipeState) -> dict:
    hist = [
        {"role": h.get("role", "user"), "content": sanitize_text(str(h.get("content", "")), MAX_HISTORY_ITEM)}
        for h in (state.get("history") or [])[-10:]
        if isinstance(h, dict)
    ]
    return {
        "goal": sanitize_text(str(state.get("goal", ""))),
        "history": hist,
        "steps": 0,
        "results": [],
        "trace": ["intake: sanitized"],
        "errors": [],
    }


def n_router(state: PipeState, boss: MainAgent, on_retry: Callable | None = None) -> dict:
    try:
        d = boss.decide(state["goal"], history=state.get("history"), stream_output=False, on_retry=on_retry)
    except Exception as e:  # noqa: BLE001 - fail closed, never fake an answer
        return {"route": "error", "final": f"Routing failed: {e}",
                "trace": ["router: ERROR fail-closed"], "errors": [f"router: {e}"],
                "steps": state.get("steps", 0) + 1}
    if d.get("type") == "chat":
        return {"route": "chat", "chat_reply": sanitize_text(d.get("reply", ""), MAX_RESULT_CHARS),
                "trace": ["router: chat"], "steps": state.get("steps", 0) + 1}
    tasks, blocked = split_tasks(d.get("tasks") or [])
    errs = [f"dispatcher: {b}" for b in blocked]
    if not tasks:
        return {"route": "error", "final": "No safe sub-tasks to run.",
                "trace": ["router: task but all blocked"], "errors": errs,
                "steps": state.get("steps", 0) + 1}
    return {"route": "task", "tasks": tasks, "remaining": list(tasks),
            "trace": [f"router: task x{len(tasks)}"], "errors": errs,
            "steps": state.get("steps", 0) + 1}


def n_chat(state: PipeState, boss: MainAgent) -> dict:
    boss.remember_exchange(state["goal"], state.get("chat_reply", ""), "chat")
    return {"final": state.get("chat_reply", ""), "trace": ["chat_answer: stored+done"]}


def n_worker(state: PipeState, on_retry: Callable | None = None) -> dict:
    from agency.worker import WorkerAgent

    remaining = list(state.get("remaining") or [])
    if not remaining:
        return {"trace": ["worker: queue empty"]}
    if state.get("steps", 0) >= STEP_BUDGET:  # budget breach: drain to synthesis
        n = len(remaining)
        return {"remaining": [], "trace": [f"worker: STEP_BUDGET hit, drained {n}"],
                "errors": [f"dispatcher: step budget exceeded, skipped {n} task(s)"]}
    t = remaining.pop(0)
    try:
        out = WorkerAgent(t["agent"]).run(
            sanitize_text(t["instruction"], MAX_CONTEXT_CHARS),
            context=_handoff(state["goal"], state.get("results") or []),
            stream_output=False, on_retry=on_retry)
        out["id"] = t["id"]
        out["output"] = sanitize_text(str(out.get("output", "")), MAX_RESULT_CHARS)
        return {"remaining": remaining, "results": [out],
                "trace": [f"worker: {t['agent']}#{t['id']} ok"],
                "steps": state.get("steps", 0) + 1}
    except Exception as e:  # noqa: BLE001 - isolate, keep pipeline alive
        return {"remaining": remaining,
                "results": [{"role": t["agent"], "id": t["id"],
                             "model": "", "output": "", "error": str(e)[:300]}],
                "trace": [f"worker: {t['agent']}#{t['id']} FAILED isolated"],
                "errors": [f"{t['agent']}#{t['id']}: {e}"],
                "steps": state.get("steps", 0) + 1}


def n_synth(state: PipeState, boss: MainAgent, on_retry: Callable | None = None) -> dict:
    good = [r for r in (state.get("results") or []) if r.get("output")]
    try:
        f = boss.synthesize(state["goal"], good or [{"role": "none", "id": 0, "output": "no worker output"}],
                            stream_output=False, on_retry=on_retry)
        final = sanitize_text(f["content"], config.DEFAULT_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001
        return {"final": "", "trace": ["synth: ERROR"],
                "errors": [f"synthesizer: {e}"]}
    boss.remember_task(state["goal"], state.get("tasks") or [], final)
    return {"final": final, "trace": ["synth: stored+done"]}


def _after_router(state: PipeState) -> Literal["chat_answer", "worker", "end"]:
    if state.get("route") == "chat":
        return "chat_answer"
    if state.get("route") == "task":
        return "worker"
    return "end"


def _after_worker(state: PipeState) -> Literal["worker", "synthesizer"]:
    return "worker" if state.get("remaining") else "synthesizer"


def build_graph(boss: MainAgent, on_retry: Callable | None = None):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(PipeState)
    g.add_node("intake", n_intake)
    g.add_node("router", lambda s: n_router(s, boss, on_retry))
    g.add_node("chat_answer", lambda s: n_chat(s, boss))
    g.add_node("worker", lambda s: n_worker(s, on_retry))
    g.add_node("synthesizer", lambda s: n_synth(s, boss, on_retry))
    g.add_edge(START, "intake")
    g.add_edge("intake", "router")
    g.add_conditional_edges("router", _after_router,
                            {"chat_answer": "chat_answer", "worker": "worker", "end": END})
    g.add_conditional_edges("worker", _after_worker,
                            {"worker": "worker", "synthesizer": "synthesizer"})
    g.add_edge("chat_answer", END)
    g.add_edge("synthesizer", END)
    return g


def run_pipeline(goal: str, history: list[dict] | None = None, thread_id: str = "default",
                 boss: MainAgent | None = None, use_memory: bool = True,
                 on_retry: Callable | None = None) -> dict:
    """Run the secure graph. Returns {mode, final, tasks, results, trace, errors}."""
    from langgraph.checkpoint.memory import MemorySaver

    boss = boss or _boss(use_memory)
    graph = build_graph(boss, on_retry).compile(checkpointer=MemorySaver())
    out: dict[str, Any] = graph.invoke(
        {"goal": goal, "history": history or []},
        config={"configurable": {"thread_id": thread_id}})
    route = out.get("route", "error")
    return {
        "goal": goal,
        "mode": "chat" if route == "chat" else "task" if route == "task" else "error",
        "final": out.get("final", ""),
        "tasks": out.get("tasks") or [],
        "results": out.get("results") or [],
        "trace": out.get("trace") or [],
        "errors": out.get("errors") or [],
    }
