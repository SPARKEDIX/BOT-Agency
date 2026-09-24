"""Main / boss agent: nvidia/nemotron-3-ultra-550b-a55b.

Smart flow (1 NIM call for chat, up to 6 for tasks):
  user msg + history -> ROUTER decides in ONE call:
    - {"type":"chat","reply":"..."}  -> answer directly, no workers
    - {"type":"task","tasks":[...]}  -> workers run SEQUENTIALLY (40 RPM safe)
                                        -> boss synthesizes final answer.
"""
import json
import re

import config
from agency import nim_client, registry
from agency.worker import WorkerAgent

ROUTER_SYSTEM = """You are the BOSS router of BOT-Agency (AI Agency).
Look at the conversation history + latest user message.

- Normal chat (greeting, small talk, simple Q&A, explanation, advice, follow-up): return {"type":"chat","reply":"..."} with the FULL reply.
- Real task (multi-step work, research + code + docs, deliverables, needs specialists): return {"type":"task","tasks":[{"id":1,"agent":"researcher|coder|coding|writer|reviewer|scraper|yt_scraper|image_maker","instruction":"..."}]} with 1-4 tasks.
Available agents: researcher, coder, coding, writer, reviewer, scraper, yt_scraper, image_maker (use scraper for any URL / fetch-website-content work, yt_scraper for YouTube channels/videos/search, coding for codegen/debugging/refactoring, image_maker for draw/generate-image/picture/photo/logo).
Keep task instructions self-contained (workers see no other history).
Return ONLY JSON, no other text."""

PLANNER_SYSTEM = """You are the BOSS planner of BOT-Agency.
Decompose the user goal into 1-4 sub-tasks.
Return ONLY a JSON array, no other text. Format:
[{"id":1,"agent":"researcher|coder|coding|writer|reviewer|scraper|yt_scraper|image_maker","instruction":"..."}]
Available agents: researcher, coder, coding, writer, reviewer, scraper, yt_scraper, image_maker (URL / website-content work, yt_scraper for YouTube, image_maker for images).
Keep instructions self-contained (workers see no other history)."""

SYNTH_SYSTEM = """You are the BOSS synthesizer of BOT-Agency.
Combine worker outputs into one final answer. Cite which agent did what.
Be direct, no fluff."""


def _validate_tasks(raw_tasks: list) -> list[dict]:
    valid_roles = set(registry.AGENTS)
    out = []
    for t in raw_tasks:
        if not isinstance(t, dict):
            continue
        agent = t.get("agent", "researcher")
        if agent not in valid_roles:
            agent = "researcher"
        instruction = str(t.get("instruction", "")).strip()
        if not instruction:
            continue
        out.append({"id": t.get("id", len(out) + 1), "agent": agent, "instruction": instruction})
    return out[:4]  # cap to protect 40 RPM budget


def _extract_json(text: str) -> list[dict]:
    """Extract first JSON array from model output (handles ```json fences)."""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("["): text.rfind("]") + 1]
    return _validate_tasks(json.loads(raw))


def _parse_router(text: str) -> dict:
    """Parse router JSON. Falls back to chat with raw text so user always gets an answer."""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    raw = m.group(1) if m else text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        # find first {...} or [...] block
        for pat in (r"(\{.*\})", r"(\[.*\])"):
            m2 = re.search(pat, text, re.S)
            if m2:
                try:
                    data = json.loads(m2.group(1))
                    break
                except Exception:
                    continue
        else:
            return {"type": "chat", "reply": text.strip() or "I didn't catch that — could you rephrase?"}
    if isinstance(data, list):  # legacy planner array -> task
        tasks = _validate_tasks(data)
        if tasks:
            return {"type": "task", "tasks": tasks}
        return {"type": "chat", "reply": text.strip()}
    if isinstance(data, dict):
        if data.get("type") == "task" and isinstance(data.get("tasks"), list):
            tasks = _validate_tasks(data["tasks"])
            if tasks:
                return {"type": "task", "tasks": tasks}
        if data.get("type") == "chat" and data.get("reply"):
            return {"type": "chat", "reply": str(data["reply"])}
    return {"type": "chat", "reply": text.strip() or "I didn't catch that — could you rephrase?"}


class MainAgent:
    def __init__(self, model: str | None = None, memory=None, use_memory: bool = True):
        self.model = model or config.MAIN_MODEL
        self._memory = memory  # AgencyMemory or None (lazy)
        self.use_memory = use_memory

    @property
    def memory(self):
        if self._memory is None and self.use_memory and config.MEMORY_ENABLED:
            try:
                from agency.memory import AgencyMemory

                self._memory = AgencyMemory()
            except Exception:
                self._memory = None
        return self._memory

    def _memory_snippet(self, query: str) -> str:
        try:
            mem = self.memory
            if mem is None or not mem.enabled:
                return ""
            return mem.recall_context(query)
        except Exception:
            return ""

    def remember_exchange(self, user_msg: str, assistant_msg: str, mode: str = "chat") -> None:
        try:
            if self.memory is not None:
                self.memory.remember_exchange(user_msg, assistant_msg, mode)
        except Exception:
            pass

    def remember_task(self, goal: str, tasks: list[dict], final: str) -> None:
        try:
            if self.memory is not None:
                self.memory.remember_task(goal, tasks, final)
        except Exception:
            pass

    def memory_stats(self) -> int:
        try:
            return self.memory.count() if self.memory is not None else 0
        except Exception:
            return 0

    def memory_clear(self) -> None:
        try:
            if self.memory is not None:
                self.memory.clear()
        except Exception:
            pass

    def decide(self, goal: str, history: list[dict] | None = None,
               stream_output: bool = False, on_retry=None) -> dict:
        """One-call router: chat (reply now) vs task (split to agents)."""
        system = ROUTER_SYSTEM
        snippet = self._memory_snippet(goal)
        if snippet:
            system += f"\n\nRelevant past memory (use for consistency):\n{snippet}"
        msgs: list[dict] = [{"role": "system", "content": system}]
        msgs += (history or [])[-10:]
        msgs.append({"role": "user", "content": goal})
        res = nim_client.chat(
            messages=msgs,
            model=self.model,
            temperature=0.3,  # deterministic routing
            top_p=0.95,
            max_tokens=4096,
            enable_thinking=False,  # router must be fast + cheap (40 RPM)
            stream_output=stream_output,
            on_retry=on_retry,
        )
        return _parse_router(res["content"])

    def plan(self, goal: str, stream_output: bool = False, on_retry=None) -> list[dict]:
        res = nim_client.chat(
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": goal},
            ],
            model=self.model,
            temperature=1.0,
            top_p=0.95,
            max_tokens=4096,
            enable_thinking=True,  # boss reasons
            stream_output=stream_output,
            on_retry=on_retry,
        )
        try:
            return _extract_json(res["content"])
        except Exception:
            # fallback: single task so agency still works
            return [{"id": 1, "agent": "writer", "instruction": goal}]

    def synthesize(self, goal: str, results: list[dict], stream_output: bool = True, on_retry=None) -> dict:
        combined = "\n\n".join(
            f"[{r['role']}#{r.get('id', i+1)}]:\n{r['output']}" for i, r in enumerate(results)
        )
        return nim_client.chat(
            messages=[
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": f"Goal:\n{goal}\n\nWorker outputs:\n{combined}"},
            ],
            model=self.model,
            temperature=1.0,
            top_p=0.95,
            max_tokens=config.DEFAULT_MAX_TOKENS,
            enable_thinking=True,
            stream_output=stream_output,
            on_retry=on_retry,
        )

    def run(self, goal: str, history: list[dict] | None = None,
            stream_output: bool = True, on_retry=None, log=print) -> dict:
        """Smart run: normal chat answered directly (1 call), tasks go to agents."""
        log(f"[main:{self.model}] routing...")
        decision = self.decide(goal, history=history, stream_output=False, on_retry=on_retry)
        if decision["type"] == "chat":
            log("[main] normal chat — answering directly, no workers.")
            self.remember_exchange(goal, decision["reply"], "chat")
            return {"goal": goal, "mode": "chat", "tasks": [],
                    "worker_results": [], "final": decision["reply"]}

        tasks = decision.get("tasks") or []
        log(f"[main] task → {len(tasks)} sub-task(s): " + ", ".join(f"{t['agent']}#{t['id']}" for t in tasks))

        results: list[dict] = []
        for t in tasks:  # sequential = never bursts 40 RPM
            log(f"[{t['agent']}] running task {t['id']}...")
            w = WorkerAgent(t["agent"])
            out = w.run(t["instruction"], context=f"Overall goal: {goal}", stream_output=False, on_retry=on_retry)
            out["id"] = t["id"]
            results.append(out)
            log(f"[{t['agent']}] done ({len(out['output'])} chars)")

        log("[main] synthesizing final answer...")
        final = self.synthesize(goal, results, stream_output=stream_output, on_retry=on_retry)
        self.remember_task(goal, tasks, final["content"])
        return {"goal": goal, "mode": "task", "tasks": tasks, "worker_results": results, "final": final["content"]}

    def direct_ask(self, prompt: str, history: list[dict] | None = None,
                   stream_output: bool = True, on_retry=None) -> dict:
        """Your base code as a method (single streaming call, thinking on)."""
        system_extra = ""
        snippet = self._memory_snippet(prompt)
        msgs: list[dict] = []
        if snippet:
            msgs.append({"role": "system", "content": f"Relevant past memory:\n{snippet}"})
        msgs += list((history or [])[-10:]) + [{"role": "user", "content": prompt}]
        return nim_client.chat(
            messages=msgs,
            model=self.model,
            temperature=1.0,
            top_p=0.95,
            max_tokens=config.DEFAULT_MAX_TOKENS,
            enable_thinking=True,
            stream_output=stream_output,
            on_retry=on_retry,
        )
