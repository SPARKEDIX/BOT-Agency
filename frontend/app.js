/* New chat — minimal client. Same-origin /api/*. No build step. */
"use strict";
const $ = (id) => document.getElementById(id);
const messagesEl = $("messages"), form = $("composer"), input = $("composer-input"),
  sendBtn = $("send-btn"), stopBtn = $("stop-btn"), errEl = $("composer-error"),
  dot = $("status-dot"), statusText = $("status-text"), banner = $("banner"),
  agentList = $("agent-list"), traceList = $("trace-list"),
  meta = $("meta"), modelLine = $("model-line");
let history = [];
let ctrl = null, timer = null, startedAt = 0;

if (location.protocol === "file:") {
  banner.hidden = false;
  banner.innerHTML = "You opened this file directly, so chat cannot reach the backend. " +
    "Start the server with <code>py backend/server.py --open</code> and use the <code>http://127.0.0.1:…</code> URL instead.";
}
const mode = () => (document.querySelector('input[name="mode"]:checked') || {}).value || "auto";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const md = (s) => esc(s).replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
  .replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>");

function setStatus(state, text) {
  dot.className = "dot" + (state ? " " + state : "");
  statusText.textContent = text;
}
function showEmpty() {
  messagesEl.innerHTML = `<p class="empty-note">Say hi, or tap a suggestion below.<br>Auto routes chat vs task · Direct is 1 call · Pipeline adds guards.</p>`;
}
function addMsg(cls, html, sub) {
  document.querySelector(".empty-note")?.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.innerHTML = html + (sub ? `<span class="meta">${esc(sub)}</span>` : "");
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return d;
}
function fail(msg) { errEl.textContent = msg; errEl.hidden = false; }
function clearFail() { errEl.hidden = true; errEl.textContent = ""; }

function setWorking(on) {
  sendBtn.hidden = on; stopBtn.hidden = !on;
  input.disabled = on;
  document.querySelectorAll(".starters button").forEach((b) => (b.disabled = on));
  clearInterval(timer);
  if (on) {
    startedAt = Date.now();
    meta.textContent = "working… 0s (NIM can take 30–90s on tasks)";
    timer = setInterval(() => { meta.textContent = `working… ${Math.round((Date.now() - startedAt) / 1000)}s — you can Cancel`; }, 1000);
  } else {
    meta.textContent = "Enter sends · Shift+Enter newline";
  }
}

async function refreshMeta() {
  try {
    const [h, a] = await Promise.all([
      fetch("/api/health").then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }),
      fetch("/api/agents").then((r) => r.json()),
    ]);
    setStatus("ok", `${h.agents} agents · ${h.memory_docs ?? 0} docs`);
    modelLine.textContent = h.boss_model || "";
    $("memory-text").textContent = `Memory: ${h.memory_docs ?? 0} docs shared.`;
    agentList.innerHTML = (a.agents || []).map((g) =>
      `<li><strong>${esc(g.role)}</strong> <code>${esc(g.model)}</code><br>${esc(g.description)}</li>`).join("");
    if (h.has_key === false) {
      banner.hidden = false;
      banner.innerHTML = "Backend has no <code>NVIDIA_API_KEY</code>. Put it in <code>.env</code> and restart: <code>py backend/server.py</code>.";
      setStatus("bad", "backend ok · key missing");
    } else if (!banner.innerHTML.includes("directly")) {
      banner.hidden = true;
    }
  } catch (e) {
    setStatus("bad", "backend unreachable");
    banner.hidden = false;
    banner.innerHTML = `Cannot reach the backend (${esc(e.message)}). Start it: <code>py backend/server.py --open</code>, then reload this page.`;
    agentList.innerHTML = `<li class="empty">Backend unreachable.</li>`;
  }
}

async function postChat(text, useMode, signal) {
  const res = await fetch("/api/chat", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message: text, history, mode: useMode}),
    signal,
  });
  const out = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(out.error || out.final || `HTTP ${res.status}`);
    err.status = res.status;
    err.payload = out;
    throw err;
  }
  return out;
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (ctrl) return;
  const text = input.value.trim();
  if (!text) { fail("Type a message first."); input.focus(); return; }
  clearFail();
  $("starters").style.display = "none";
  addMsg("user", md(text));
  input.value = "";
  autoGrow();
  const bubble = addMsg("assistant loading", "working…", mode());
  setStatus("working", "working…");
  setWorking(true);
  ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort("timeout"), 180000);
  try {
    let out, usedMode = mode();
    try {
      out = await postChat(text, usedMode, ctrl.signal);
    } catch (e) {
      // Easy pipeline: NIM overload on multi-call modes → retry once as Direct (1 call).
      if ((e.status === 502 || e.status === 500) && usedMode !== "direct") {
        bubble.innerHTML = `Busy — retrying as Direct (1 call)…<span class="meta">${esc(usedMode)} → direct</span>`;
        out = await postChat(text, "direct", ctrl.signal);
        usedMode = "direct (fallback)";
      } else {
        throw e;
      }
    }
    bubble.classList.remove("loading");
    const tasks = (out.tasks || []).map((t) => `#${t.id} ${t.agent}`).join(", ");
    bubble.innerHTML = md(out.final || "_empty response_") + `<span class="meta">${esc([out.mode || usedMode, tasks].filter(Boolean).join(" · "))}</span>`;
    if (out.trace?.length) traceList.innerHTML = out.trace.map((t) => `<li>${esc(t)}</li>`).join("");
    if (out.errors?.length) bubble.innerHTML += `<span class="meta">⚠ ${esc(out.errors.join("; "))}</span>`;
    history.push({role: "user", content: text}, {role: "assistant", content: out.final || ""});
    history = history.slice(-20);
    refreshMeta();
  } catch (e) {
    bubble.className = "msg error";
    const hint = e.name === "AbortError"
      ? "Request cancelled or timed out (3 min). NIM may be overloaded — retry with Direct mode."
      : `${e.message}. Check the server terminal, then retry.`;
    bubble.innerHTML = `<strong>Request failed.</strong><br>${esc(hint)}`;
    fail(e.message);
  } finally {
    clearTimeout(timeout); ctrl = null;
    setWorking(false); setStatus("ok", statusText.textContent.replace("working…", "ready"));
    messagesEl.scrollTop = messagesEl.scrollHeight;
    input.focus();
  }
});
stopBtn.addEventListener("click", () => ctrl?.abort("cancelled"));
document.querySelectorAll(".starters button").forEach((b) =>
  b.addEventListener("click", () => { input.value = b.dataset.q; autoGrow(); form.requestSubmit(); }));
$("memory-clear").addEventListener("click", async () => {
  await fetch("/api/memory/clear", {method: "POST"}); refreshMeta();
});
function autoGrow() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; }
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
showEmpty();
refreshMeta();
