"""AI Agency — Claude Code style CLI. NIM-only boss + workers.

Single-shot:
  py main.py "Write a limerick about GPUs" --direct
  py main.py "Research X, write code, summarize"
  py main.py --interactive   (force REPL)

REPL slash commands:
  /help /agents /model /auto /direct /agency /pipeline /stream /memory /clear /quit
"""
import argparse
import shutil
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import config
from agency.main_agent import MainAgent
from agency import registry
from agency.worker import WorkerAgent

console = Console()
APP = "AI Agency"

# Single source of truth for slash commands: Tab-completion, /help, and parsing.
SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help",
    "/agents": "List worker agents in the ecosystem",
    "/model": "Show or switch boss model — /model <NIM id>",
    "/auto": "Smart mode (default): chat replies directly, tasks split to agents",
    "/direct": "Force single boss call (1 request, best when NIM is busy)",
    "/agency": "Force full agency (plan → workers → synthesize)",
    "/pipeline": "Secure LangGraph run (blackboard, trace, guards)",
    "/stream": "Toggle token streaming on/off",
    "/memory": "Show Chroma docs count — /memory clear wipes memory",
    "/clear": "Clear screen",
    "/quit": "Exit (also /exit, /q)",
}


def read_line() -> str:
    """Prompt with Tab-completion for /commands (prompt_toolkit), fallback to plain."""
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.completion import WordCompleter

        completer = WordCompleter(list(SLASH_COMMANDS), ignore_case=True, sentence=True)
        return pt_prompt([("class:prompt", "❯ ")], completer=completer).strip()
    except Exception:
        return Prompt.ask("\n[bold cyan]❯[/bold cyan]").strip()


def suggest_commands(partial: str) -> list[str]:
    """Close matches for an unknown /command (difflib + prefix)."""
    import difflib

    names = list(SLASH_COMMANDS)
    if partial in ("", "/"):
        return names
    cands = [n for n in names if n.startswith(partial)]
    cands += [c for c in difflib.get_close_matches(partial, names, n=3, cutoff=0.5) if c not in cands]
    return cands


def show_suggestions(partial: str) -> None:
    cands = suggest_commands(partial)
    if not cands:
        console.print("[yellow]No matching command. Try /help.[/]")
        return
    t = Table(show_header=False, box=None)
    t.add_column("Command", style="green")
    t.add_column("What it does", style="dim")
    for c in cands:
        t.add_row(c, SLASH_COMMANDS[c])
    console.print(Panel(t, title=f"did you mean (for '{partial}')?" if partial != "/" else "commands", border_style="cyan"))


def banner(boss_model: str, mode: str, stream: bool, mem_count: int | None = None) -> None:
    width = min(shutil.get_terminal_size((80, 20)).columns, 90)
    mem_line = f"  [dim]• memory:[/] {mem_count} docs" if mem_count is not None else ""
    console.print(
        Panel.fit(
            f"[bold cyan]{APP}[/]  [dim]NIM-only agent agency + Chroma memory[/]\n"
            f"[dim]boss:[/] [green]{boss_model}[/]\n"
            f"[dim]mode:[/] {mode}  [dim]• stream:[/] {'on' if stream else 'off'}  "
            f"[dim]• budget:[/] 40 RPM shared{mem_line}",
            border_style="cyan",
            padding=(0, 2),
        ),
        width=width,
    )
    console.print("[dim]Just chat normally — tasks are auto-split to agents. /help for commands.[/]\n")


def show_help() -> None:
    t = Table(show_header=True, header_style="bold cyan", box=None)
    t.add_column("Command", style="green")
    t.add_column("What it does", style="white")
    for cmd, desc in SLASH_COMMANDS.items():
        t.add_row(cmd, desc)
    console.print(Panel(t, title="commands", border_style="cyan"))


def show_agents() -> None:
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Agent", style="green")
    t.add_column("Model", style="dim")
    t.add_column("Description", style="white")
    for role, info in registry.AGENTS.items():
        t.add_row(role, registry.model_for(role), info["description"])
    console.print(Panel(t, title="ecosystem (all NIM, one key)", border_style="cyan"))


def _on_retry(attempt: int, total: int, msg: str) -> None:
    console.print(f"[yellow]↻ NIM busy — retry {attempt}/{total}: {msg}[/yellow]")


def _task_table(tasks: list[dict]) -> Table:
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("#", style="dim", width=4)
    t.add_column("Agent", style="green")
    t.add_column("Instruction", style="white")
    for task in tasks:
        t.add_row(str(task["id"]), task["agent"], task["instruction"][:120])
    return t


def show_memory(boss: MainAgent) -> None:
    try:
        n = boss.memory_stats()
    except Exception:
        n = 0
    console.print(Panel(f"[green]{n}[/] docs in Chroma [dim]({config.CHROMA_PATH}, top_k={config.MEMORY_TOP_K})[/]\n"
                        "[dim]Use /memory clear to wipe.[/]", title="vector memory", border_style="magenta"))


def run_direct(goal: str, boss: MainAgent, history: list[dict], stream: bool) -> str:
    with console.status("[cyan]Boss thinking… (1 NIM call)[/]", spinner="dots"):
        # run without inner streaming; we render markdown after for clean UI
        res = boss.direct_ask(goal, history=history, stream_output=False, on_retry=_on_retry)
    console.print(Panel(Markdown(res["content"] or "_empty response_"), title="boss answer", border_style="green"))
    boss.remember_exchange(goal, res["content"], "direct")
    return res["content"]


def run_auto(goal: str, boss: MainAgent, history: list[dict], stream: bool) -> str:
    """Smart: router decides chat (1 call) vs task (workers + synth)."""
    with console.status("[cyan]Boss routing… (chat or task?)[/]", spinner="dots"):
        decision = boss.decide(goal, history=history, stream_output=False, on_retry=_on_retry)
    if decision["type"] == "chat":
        console.print(Panel(Markdown(decision["reply"] or "_empty response_"),
                            title="boss answer", subtitle="[dim]direct — no workers[/]", border_style="green"))
        boss.remember_exchange(goal, decision["reply"], "chat")
        return decision["reply"]

    tasks = decision["tasks"]
    console.print(Panel(_task_table(tasks), title=f"task split — {len(tasks)} sub-task(s)", border_style="cyan"))
    results: list[dict] = []
    for task in tasks:  # sequential: never bursts 40 RPM
        with console.status(f"[cyan]{task['agent']} #{task['id']} working…[/]", spinner="dots"):
            w = WorkerAgent(task["agent"])
            out = w.run(task["instruction"], context=f"Overall goal: {goal}",
                        stream_output=False, on_retry=_on_retry)
            out["id"] = task["id"]
            results.append(out)
        console.print(f"[green]✓[/] [bold]{task['agent']}[/] #{task['id']} done ({len(out['output'])} chars)")

    with console.status("[cyan]Boss synthesizing final answer…[/]", spinner="dots"):
        final = boss.synthesize(goal, results, stream_output=False, on_retry=_on_retry)
    console.print(Panel(Markdown(final["content"] or "_empty response_"), title="final answer", border_style="green"))
    boss.remember_task(goal, tasks, final["content"])
    return final["content"]


def run_agency(goal: str, boss: MainAgent, history: list[dict], stream: bool) -> str:
    with console.status("[cyan]Boss planning sub-tasks…[/]", spinner="dots"):
        tasks = boss.plan(goal, stream_output=False, on_retry=_on_retry)
    console.print(Panel(_task_table(tasks), title=f"plan — {len(tasks)} sub-task(s)", border_style="cyan"))

    results: list[dict] = []
    for task in tasks:  # sequential: never bursts 40 RPM
        with console.status(f"[cyan]{task['agent']} #{task['id']} working…[/]", spinner="dots"):
            w = WorkerAgent(task["agent"])
            out = w.run(task["instruction"], context=f"Overall goal: {goal}",
                        stream_output=False, on_retry=_on_retry)
            out["id"] = task["id"]
            results.append(out)
        console.print(f"[green]✓[/] [bold]{task['agent']}[/] #{task['id']} done ({len(out['output'])} chars)")

    with console.status("[cyan]Boss synthesizing final answer…[/]", spinner="dots"):
        final = boss.synthesize(goal, results, stream_output=False, on_retry=_on_retry)
    console.print(Panel(Markdown(final["content"] or "_empty response_"), title="final answer", border_style="green"))
    boss.remember_task(goal, tasks, final["content"])
    return final["content"]


def run_pipeline_mode(goal: str, boss: MainAgent, history: list[dict]) -> str:
    """Secure LangGraph run: shared blackboard, traced, fail-closed."""
    from agency.pipeline import run_pipeline

    with console.status("[cyan]Pipeline running… (router → agents → synth)[/]", spinner="dots"):
        out = run_pipeline(goal, history=history, boss=boss, on_retry=_on_retry)
    if out["trace"]:
        console.print(Panel("\n".join(f"[dim]•[/] {t}" for t in out["trace"]),
                            title="pipeline trace", border_style="cyan"))
    if out["errors"]:
        console.print(Panel("\n".join(f"[yellow]•[/] {e}" for e in out["errors"]),
                            title="blocked / failed (isolated)", border_style="yellow"))
    title = {"chat": "boss answer", "task": "final answer"}.get(out["mode"], "pipeline error")
    console.print(Panel(Markdown(out["final"] or "_empty response_"), title=title, border_style="green"))
    return out["final"]


def handle_goal(goal: str, boss: MainAgent, mode: str, history: list[dict], stream: bool) -> None:
    try:
        if mode == "direct":
            reply = run_direct(goal, boss, history, stream)
        elif mode == "agency":
            reply = run_agency(goal, boss, history, stream)
        elif mode == "pipeline":
            reply = run_pipeline_mode(goal, boss, history)
        else:  # auto/smart (default)
            reply = run_auto(goal, boss, history, stream)
        history.append({"role": "user", "content": goal})
        history.append({"role": "assistant", "content": reply})
        del history[:-20]  # keep last 20 turns
    except RuntimeError as e:
        console.print(Panel(
            f"[red]{e}[/]\n\n[dim]NIM said 'Service temporarily overloaded'. This is server-side, not your code.\n"
            "• Wait 30–60s and retry (auto-retries already ran 5× with backoff)\n"
            "• Use /direct (1 request) instead of /agency (up to 6 requests)\n"
            "• Check key in .env and status at build.nvidia.com[/]",
            title="NIM request failed", border_style="red",
        ))
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/]")


def repl(boss: MainAgent, mode: str, stream: bool) -> int:
    try:
        banner(boss.model, mode, stream, boss.memory_stats())
    except Exception:
        banner(boss.model, mode, stream)
    history: list[dict] = []
    while True:
        try:
            line = read_line()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
            if cmd == "/":
                show_suggestions("/")
                continue
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Bye.[/]")
                return 0
            elif cmd == "/help":
                show_help()
            elif cmd == "/agents":
                show_agents()
            elif cmd == "/memory":
                if arg.strip().lower() == "clear":
                    boss.memory_clear()
                    console.print("[green]Memory cleared.[/]")
                else:
                    show_memory(boss)
            elif cmd == "/model":
                if arg:
                    boss.model = arg.strip()
                    console.print(f"[green]Boss model → {boss.model}[/]")
                else:
                    console.print(f"[dim]boss:[/] {boss.model}")
            elif cmd == "/auto":
                mode = "auto"
                console.print("[green]Mode → auto (chat directly, split tasks)[/]")
            elif cmd == "/direct":
                mode = "direct"
                console.print("[green]Mode → direct (1 NIM call)[/]")
            elif cmd == "/agency":
                mode = "agency"
                console.print("[green]Mode → agency (forced plan → workers → synthesize)[/]")
            elif cmd == "/pipeline":
                mode = "pipeline"
                console.print("[green]Mode → pipeline (secure LangGraph: blackboard + trace)[/]")
            elif cmd == "/stream":
                stream = not stream
                console.print(f"[green]Stream → {'on' if stream else 'off'}[/]")
            elif cmd == "/clear":
                console.clear()
                try:
                    banner(boss.model, mode, stream, boss.memory_stats())
                except Exception:
                    banner(boss.model, mode, stream)
            else:
                show_suggestions(cmd)
            continue
        handle_goal(line, boss, mode, history, stream)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AI Agency — NIM-only boss + workers (Claude-style CLI)")
    p.add_argument("goal", nargs="*", help="Goal to run once (empty = interactive REPL)")
    p.add_argument("--direct", action="store_true", help="Force single boss call (1 request)")
    p.add_argument("--agency", action="store_true", help="Force full agency run")
    p.add_argument("--pipeline", action="store_true", help="Secure LangGraph pipeline run")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming")
    p.add_argument("--interactive", "-i", action="store_true", help="Force REPL after single-shot")
    p.add_argument("--model", default=config.MAIN_MODEL, help="Boss NIM model id")
    p.add_argument("--no-memory", action="store_true", help="Disable Chroma vector memory")
    p.add_argument("--debug", action="store_true", help="Show full tracebacks")
    args = p.parse_args(argv)

    mode = "direct" if args.direct else "agency" if args.agency else "pipeline" if args.pipeline else "auto"
    stream = not args.no_stream
    boss = MainAgent(model=args.model, use_memory=not args.no_memory)
    goal = " ".join(args.goal).strip()

    if not goal:
        return repl(boss, mode, stream)

    # single-shot
    banner(boss.model, mode, stream)
    console.print(f"[bold cyan]❯[/] {goal}\n")
    try:
        handle_goal(goal, boss, mode, [], stream)
    except Exception:
        if args.debug:
            raise
        console.print("[red]Failed. Re-run with --debug or try /direct + wait 60s.[/]")
        return 1
    if args.interactive:
        return repl(boss, mode, stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
