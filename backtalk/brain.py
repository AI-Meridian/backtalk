# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

try:
    from claude_agent_sdk import CanUseToolShadowedWarning
except ImportError:                       # older SDKs: nothing to silence
    CanUseToolShadowedWarning = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.path_guard import guard_blocked_paths
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

# Plain-English labels for a face to show while a tool runs and the
# brain has gone quiet. Unlisted tools (custom MCP tools, future
# built-ins) still get a readable fallback rather than showing nothing.
_TOOL_LABELS = {
    "Bash": "Running a command",
    "Read": "Reading a file",
    "Write": "Writing a file",
    "Edit": "Editing a file",
    "NotebookEdit": "Editing a notebook",
    "Glob": "Searching for files",
    "Grep": "Searching for text",
    "WebSearch": "Searching the web",
    "WebFetch": "Reading a web page",
    "Agent": "Delegating to a subagent",
    "Task": "Delegating to a subagent",
    "TodoWrite": "Updating the task list",
    "SlashCommand": "Running a command",
}


_MAX_LABEL_LEN = 72


def _truncate(s: str, limit: int = _MAX_LABEL_LEN) -> str:
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _task_label(tool_name: str, tool_input: dict | None = None) -> str:
    """A specific, real-time label built from this exact call's own
    input, not a generic per-tool-type phrase — the task panel should
    read like a live line of what's actually happening (the real file,
    the real command, the real search term), the same way Claude Code's
    own terminal output does, not a static word that never changes
    across an entire turn."""
    ti = tool_input or {}
    if tool_name == "Read":
        p = ti.get("file_path", "")
        return f"Reading {Path(p).name}" if p else _TOOL_LABELS["Read"]
    if tool_name == "Write":
        p = ti.get("file_path", "")
        return f"Writing {Path(p).name}" if p else _TOOL_LABELS["Write"]
    if tool_name == "Edit":
        p = ti.get("file_path", "")
        return f"Editing {Path(p).name}" if p else _TOOL_LABELS["Edit"]
    if tool_name == "NotebookEdit":
        p = ti.get("notebook_path", "")
        return f"Editing {Path(p).name}" if p else _TOOL_LABELS["NotebookEdit"]
    if tool_name == "Bash":
        cmd = (ti.get("description") or ti.get("command") or "").strip()
        return _truncate(cmd) if cmd else _TOOL_LABELS["Bash"]
    if tool_name == "Glob":
        pat = ti.get("pattern", "")
        return _truncate(f"Searching for {pat}") if pat else _TOOL_LABELS["Glob"]
    if tool_name == "Grep":
        pat = ti.get("pattern", "")
        return _truncate(f"Searching for ‘{pat}’") if pat else _TOOL_LABELS["Grep"]
    if tool_name == "WebSearch":
        q = ti.get("query", "")
        return _truncate(f"Searching the web for ‘{q}’") if q else _TOOL_LABELS["WebSearch"]
    if tool_name == "WebFetch":
        url = ti.get("url", "")
        return _truncate(f"Reading {url}") if url else _TOOL_LABELS["WebFetch"]
    if tool_name in ("Agent", "Task"):
        desc = ti.get("description", "")
        return _truncate(f"Delegating: {desc}") if desc else _TOOL_LABELS[tool_name]
    if tool_name == "SlashCommand":
        cmd = ti.get("command", "")
        return _truncate(f"Running {cmd}") if cmd else _TOOL_LABELS["SlashCommand"]
    return _TOOL_LABELS.get(tool_name, f"Working: {tool_name}" if tool_name
                             else "Working")


def _task_eta(tool_name: str, tool_input: dict) -> float | None:
    """Best-effort duration estimate, seconds, for a face to render a
    real progress bar against — not a guess invented here, just the
    timeout the tool call itself already carries. Bash is the only
    built-in tool with a caller-set duration budget (ms); the SDK's own
    documented default is 120000ms when the caller didn't pass one."""
    if tool_name != "Bash":
        return None
    timeout_ms = tool_input.get("timeout") or 120_000
    return timeout_ms / 1000.0


async def _task_start_hook(input_data, tool_use_id, context):
    """PreToolUse hook: mark this call as running the moment the SDK
    actually dispatches it for execution, keyed by the SDK's own
    tool_use_id rather than guessed from the raw content-block stream.
    Parallel tool calls each get their own id, so they show up as
    independent entries instead of racing to overwrite one slot."""
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input") or {}
    signals.start_task(tool_use_id, _task_label(tool_name, tool_input),
                        eta=_task_eta(tool_name, tool_input))
    return {}


async def _task_end_hook(input_data, tool_use_id, context):
    """PostToolUse / PostToolUseFailure hook: that call is done (or
    failed) either way — drop it from the active set."""
    signals.end_task(tool_use_id)
    return {}


# Registered unconditionally, in every permission_mode including
# bypassPermissions — see path_guard.py for why this has to be a
# PreToolUse hook rather than living in the can_use_tool gate below.
# The task-tracking hooks ride the same mechanism for the same reason:
# a hook fires for every tool call regardless of permission_mode, which
# a purely stream-based guess (the old approach) can't rely on. An empty
# config.blocked_paths makes guard_blocked_paths itself a no-op per
# call, so none of this costs anything extra when unused.
_HOOKS = {
    "PreToolUse": [HookMatcher(matcher=None,
                                hooks=[guard_blocked_paths, _task_start_hook])],
    "PostToolUse": [HookMatcher(matcher=None, hooks=[_task_end_hook])],
    "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[_task_end_hook])],
}

# How long ask_stream will wait for ANY stream activity (not the whole
# turn — a heavy tool call, e.g. reviewing dozens of image frames, can
# legitimately go quiet for a while mid-turn) before presuming the CLI
# subprocess itself has stalled and rebuilding rather than hanging
# forever. Field case: a 40-frame video review choked the pipe, and
# ask_stream had no bound at all, so the voice line just sat dead.
_ASK_IDLE_TIMEOUT = 240

# The SDK's subprocess transport caps a single JSON stdout message at
# 1MB by default and kills the pipe past that ("JSON message exceeded
# maximum buffer size"), which reset_turn can't recover from — it reads
# as a dead pipe and rebuilds the whole session, wiping conversation
# memory. A single oversized tool result (a big ptt payload, a large
# image/doc read) can trip that default. 10x headroom.
_MAX_BUFFER_SIZE = 10 * 1024 * 1024


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # The spoken permission gate (main.py builds it). Wired at
        # connect in EVERY mode, so a live mode flip needs no reconnect;
        # bypass simply never consults it.
        self._can_use_tool = can_use_tool
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the FIRST start only (config key
        # resume_last_session). Consumed on use: a desync rebuild in
        # reset_turn() must always start FRESH: a rebuild means a turn
        # went sideways mid-stream, the wrong moment to gamble on
        # reattaching. (Community proposal, issue #1.)
        self._resume_id = resume_id
        # True while a query's response hasn't been consumed through its
        # ResultMessage — i.e. the shared message pipe may hold leftovers.
        self._dirty = False

    async def start(self):
        mode = CFG["permission_mode"]
        if mode == "default":
            mode = "ask"     # legacy alias, see config.py
        # backtalk's "ask" AND "confirm_risk" both = the SDK's "default"
        # mode, gated calls routed to the can_use_tool gate either way —
        # "confirm_risk" still needs every gated call to reach the gate,
        # it just decides per-call there whether to ask or wave it
        # through, instead of asking every time like "ask" does.
        sdk_mode = "default" if mode in ("ask", "confirm_risk") else mode
        if sdk_mode == "bypassPermissions" and self._can_use_tool \
                and CanUseToolShadowedWarning:
            # Deliberate auto-approve: the SDK warns that the callback is
            # shadowed. That IS the chosen behavior, so boot quietly.
            warnings.filterwarnings("ignore",
                                    category=CanUseToolShadowedWarning)
        resume, self._resume_id = self._resume_id, None   # consume once

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": DISCIPLINE},
                include_partial_messages=True,
                permission_mode=sdk_mode,
                can_use_tool=self._can_use_tool,
                add_dirs=CFG["extra_dirs"],
                skills=CFG["visible_skills"],
                resume=rid,
                max_buffer_size=_MAX_BUFFER_SIZE,
                hooks=_HOOKS,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                log(f"[brain] resumed session {resume[:8]}")
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        self._client = ClaudeSDKClient(options=_opts(None))
        await self._client.connect()

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip, no reconnect, conversation intact ("ask" maps to
        the SDK's "default", whose gated calls hit the spoken gate)."""
        if self._client:
            sdk_mode = "default" if backtalk_mode == "ask" \
                else backtalk_mode
            await self._client.set_permission_mode(sdk_mode)

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Must never
        break a turn; silence on any failure."""
        if not CFG.get("resume_last_session"):
            return
        sid = getattr(rm, "session_id", None)
        if not sid:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
        except OSError:
            pass

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                signals.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) through the normal stream and return whatever text the
        CLI answered with (confirmations, errors). Slash-command replies
        arrive as COMPLETE AssistantMessages, not stream deltas, so
        ask_stream cannot see them. Bounded like reset_turn is: this
        stream is not trusted to always deliver, and an unbounded await
        here would deafen the whole voice loop. On timeout the pipe is
        left marked dirty so the next reset_turn drains or rebuilds."""
        self._dirty = True
        await self._client.query(cmd)
        texts = []

        async def _collect():
            async for msg in self._client.receive_response():
                t = type(msg).__name__
                if t == "AssistantMessage":
                    for b in getattr(msg, "content", []) or []:
                        txt = getattr(b, "text", None)
                        if txt:
                            texts.append(txt)
                elif t == "ResultMessage":
                    self._dirty = False
                    self._tally(msg, count_turn=False)
                    self._remember_session(msg)
                    break

        try:
            await asyncio.wait_for(_collect(), 90)
        except asyncio.TimeoutError:
            log(f"[brain] console command timed out: {cmd!r}")
            return "error: the command timed out"
        return " ".join(texts).strip()

    async def interrupt(self):
        if self._client:
            await self._client.interrupt()

    async def reset_turn(self, timeout: float = 8.0):
        """Re-align the message pipe after an interrupted/failed turn.

        THE OFF-BY-ONE BUG, and why this method exists: the SDK client
        has ONE shared message stream and receive_response() stops at
        the FIRST ResultMessage it sees — there is no pairing between a
        query and its response. A cancelled turn stops consuming
        mid-stream, leaving the dead turn's remaining messages
        (including its ResultMessage) buffered. The next query then
        pairs with those leftovers: the first ask lands on the stale
        ResultMessage and yields nothing, and every ask after that
        answers the PREVIOUS question — for the rest of the session.
        So: interrupt the dead turn, then drain the pipe through its
        stale ResultMessage before the next query goes out. No-op when
        the last turn was consumed clean."""
        if not self._client or not self._dirty:
            return
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # turn may already be over — the drain below is the point

        async def _drain() -> int:
            n = 0
            async for msg in self._client.receive_response():
                n += 1
                if type(msg).__name__ == "ResultMessage":
                    break
            return n

        try:
            drained = await asyncio.wait_for(_drain(), timeout)
            if drained == 0:
                # A genuinely resynced pipe always yields at least the
                # dead turn's leftover ResultMessage (per the docstring
                # above). Zero means there was nothing left to find
                # because the subprocess connection itself is gone, not
                # that the pipe happened to already be aligned — every
                # healthy interrupt in the field logs drains 2+ messages,
                # never 0. Fall through to the same rebuild as a timeout.
                raise RuntimeError("drain returned zero messages — "
                                    "pipe is dead, not just desynced")
            log(f"[brain] interrupted turn drained ({drained} stale messages)")
            self._dirty = False
        except Exception:
            # Can't re-align — rebuild the session rather than run
            # desynced. Loses this voice session's conversation memory;
            # better than answering every question one turn late for the
            # rest of the day.
            log("[brain] stream desynced beyond repair — rebuilding the "
                "session (conversation memory for this session resets)")
            await self._rebuild_session()

    async def stop(self):
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def _rebuild_session(self):
        """Disconnect and reconnect fresh. Loses this voice session's
        conversation memory; called only once the pipe to the CLI
        subprocess is confirmed dead, never for a routine resync."""
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        await self.start()
        self._dirty = False

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model.

        Bounded by _ASK_IDLE_TIMEOUT per message, not per turn: a slow
        but ALIVE tool call (a big image review, a web fetch) can go
        quiet for a while and that's fine, but a subprocess that never
        produces another message again is a dead pipe, and this used to
        have no bound at all — a stalled turn just hung the voice line
        forever with no recovery short of killing the process."""
        self._dirty = True             # in flight until its ResultMessage
        await self._client.query(utterance)
        buf = ""
        stream = self._client.receive_response()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(stream.__anext__(),
                                                  _ASK_IDLE_TIMEOUT)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    log(f"[brain] ask_stream stalled ({_ASK_IDLE_TIMEOUT}s "
                        "with no stream activity) — rebuilding the session")
                    await self._rebuild_session()
                    yield ("Sorry, I lost the connection mid-turn there. "
                           "I've reconnected — go ahead and ask again.")
                    return
                t = type(msg).__name__
                if t == "StreamEvent":
                    ev = getattr(msg, "event", {}) or {}
                    etype = ev.get("type")
                    if etype == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        if delta.get("type") == "text_delta":
                            buf += delta.get("text", "")
                            # emit any complete sentences
                            while True:
                                m = _SENTENCE_END.search(buf)
                                if not m:
                                    break
                                sentence, buf = (buf[:m.end()].strip(),
                                                 buf[m.end():])
                                if sentence:
                                    yield sentence
                    elif etype == "content_block_stop":
                        # End of a speech block (e.g. right before a tool
                        # call): flush NOW. Without this, pre-tool filler
                        # ("On it — let me grab that.") sits silent in the
                        # buffer through the whole tool run, then plays
                        # glued to the answer: long dead air, then two
                        # thoughts at once.
                        tail = buf.strip()
                        buf = ""
                        if tail:
                            yield tail
                elif t == "ResultMessage":
                    self._dirty = False    # turn fully consumed — pipe aligned
                    self._tally(msg)
                    self._remember_session(msg)
                    await self._pull_rate_limits()
                    break
            tail = buf.strip()
            if tail:
                yield tail
        finally:
            # Always clear, even on a stall/rebuild or an exception mid-turn
            # — a stuck "Running a command" is worse than showing nothing.
            # The hooks above should have already removed every task that
            # finished cleanly; this is only the safety net for one that
            # didn't (a killed subprocess, a dropped PostToolUse event).
            signals.clear_all_tasks()


if __name__ == "__main__":
    import time

    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())
