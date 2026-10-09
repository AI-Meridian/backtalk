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
"""The signal bus — tiny files any other program can watch.

The voice line leaves notes; faces read the notes. That one dumb trick
is the whole integration surface:

  .voice_state        idle | listening | thinking | speaking
  .voice_waveform     JSON {ts, samples: [64 floats]} while audio plays
  .voice_input_waveform  JSON {ts, samples: [64 floats]} while the mic
                      is open and listening for an utterance
  .voice_loading_pid  exists while the thinking sound is playing
  .voice_rate_limits  JSON {window: {utilization, resets_at}} — only
                      written when show_usage is on
  .voice_task         JSON list of {id, ts, label, eta} — one entry per
                      tool call currently executing. Several entries at
                      once means several tools are genuinely running in
                      parallel, not a display glitch. Empty/absent means
                      nothing running right now. `eta` (seconds, may be
                      null) lets a face show a real progress bar for a
                      long job instead of just an elapsed-time spinner.
  .voice_task_completed  JSON {id, label, completed_ts} — the most
                      recently FINISHED task (tool call or background
                      agent), kept around after it leaves .voice_task
                      so a face can show a persistent "last done" row
                      instead of a result that just vanishes. Overwritten
                      by the next completion, never cleared — once
                      something has finished this run, this file always
                      names the newest one to finish.

Written to signals_dir (default: the repo root). Visualizers built on
this contract just work.

THE BAREHANDS SEAM: set barehands_state_dir in backtalk.json to a
barehands checkout's state/ folder and the same signals are mirrored in
its format (state/state as a bare word, state/wave.json normalized
0..1) — the on-screen ring becomes your agent's face with zero glue.

Every write is wrapped: the bus must never crash the voice line.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from backtalk.config import CFG

_DIR = CFG["signals_dir"]
_STATE_FILE = os.path.join(_DIR, ".voice_state")
_WAVEFORM_FILE = os.path.join(_DIR, ".voice_waveform")
_INPUT_WAVEFORM_FILE = os.path.join(_DIR, ".voice_input_waveform")
_LOADING_PID_FILE = os.path.join(_DIR, ".voice_loading_pid")
_DIRECTION_FILE = os.path.join(_DIR, ".voice_direction")
_REPLY_DONE_FILE = os.path.join(_DIR, ".voice_reply_done")
_RATE_LIMIT_FILE = os.path.join(_DIR, ".voice_rate_limits")
_TASK_FILE = os.path.join(_DIR, ".voice_task")
_LAST_DONE_FILE = os.path.join(_DIR, ".voice_task_completed")

_BH = CFG.get("barehands_state_dir") or ""
_BH_STATE = os.path.join(_BH, "state") if _BH else ""
_BH_WAVE = os.path.join(_BH, "wave.json") if _BH else ""

_THINKING_SOUND = CFG.get("thinking_sound") or ""

_WAVEFORM_MIN_INTERVAL = 1.0 / 15   # ~15 writes/sec is plenty for 60fps reads
_last_waveform_write = 0.0
_last_input_waveform_write = 0.0
_static_proc: subprocess.Popen | None = None


def set_state(name: str):
    """Write the state. Never raises — the show must go on."""
    try:
        with open(_STATE_FILE, "w") as f:
            f.write(name)
    except OSError:
        pass
    if _BH_STATE:
        try:
            with open(_BH_STATE, "w") as f:
                f.write(name)
        except OSError:
            pass


# The active set of tool calls, keyed by the SDK's own tool_use_id so
# calls that genuinely overlap (parallel tool use in one turn) each get
# tracked independently instead of clobbering a single shared slot —
# that clobbering was the real bug behind the old "task panel isn't
# consistently on" complaint: a second tool starting before the first
# one's content_block_stop fired would just overwrite the first's label.
_tasks: dict[str, dict] = {}

# Background agents (SubagentStart/SubagentStop), kept in a separate set
# from _tasks on purpose: their real lifetime is bounded by their own
# start/stop pair, not by this turn's — clear_all_tasks() runs at the
# end of EVERY turn (including the one that just launched a background
# agent and kept talking), and must not wipe a job that is deliberately
# still running after the voice line has gone quiet again.
_subagent_tasks: dict[str, dict] = {}
_heartbeat_thread: threading.Thread | None = None
_HEARTBEAT_INTERVAL = 10.0   # well under ai-visualizer's 30s stale_limit

# The single most recent completion (tool call or background agent),
# kept after its entry leaves _tasks/_subagent_tasks so a face can show
# a persistent "last done" row. Deliberately a bare dict, not a list —
# one real result, replaced by the next one, never an accreting log.
_last_done: dict | None = None


def _write_last_done():
    if _last_done is None:
        return
    try:
        with open(_LAST_DONE_FILE, "w") as f:
            json.dump(_last_done, f)
    except OSError:
        pass


def _record_done(entry: dict):
    global _last_done
    _last_done = {"id": entry["id"], "label": entry["label"],
                  "completed_ts": time.time()}
    _write_last_done()


def _write_tasks():
    try:
        merged = list(_tasks.values()) + list(_subagent_tasks.values())
        if merged:
            with open(_TASK_FILE, "w") as f:
                json.dump(merged, f)
        else:
            os.remove(_TASK_FILE)
    except OSError:
        pass


def start_task(task_id: str, label: str, eta: float | None = None):
    """A tool call started executing — add it to the active set, in
    plain English ("Reading a file"), for a face to show while the
    brain is quietly working instead of talking. `eta` (seconds) lets a
    long job carry a real progress estimate; omit it for anything whose
    duration isn't known up front. Never raises."""
    _tasks[task_id] = {"id": task_id, "ts": time.time(), "label": label,
                        "eta": eta}
    _write_tasks()


def end_task(task_id: str):
    """That tool call finished (or failed) — drop it from the active
    set and record it as the newest completion. Never raises."""
    entry = _tasks.pop(task_id, None)
    if entry is not None:
        _record_done(entry)
    _write_tasks()


def clear_all_tasks():
    """Safety net for a stall/rebuild/exception mid-turn: wipe every
    active TOOL-CALL task rather than leave a stuck entry on screen
    forever. Deliberately leaves _subagent_tasks untouched — a
    background agent survives past the turn that launched it."""
    _tasks.clear()
    _write_tasks()


def _heartbeat_loop():
    """Keeps every live background-agent row looking fresh to
    ai-visualizer's own staleness check, which only knows how to judge
    a tool call's normal few-second lifetime (or a Bash call's explicit
    eta) — not a multi-minute agent run with no duration estimate at
    all. Touches `hb` only, never `ts`, so a face's elapsed-time display
    (built off `ts`) keeps reporting the real time since the agent
    actually started, not a reset clock."""
    while True:
        time.sleep(_HEARTBEAT_INTERVAL)
        if not _subagent_tasks:
            continue
        now = time.time()
        for entry in _subagent_tasks.values():
            entry["hb"] = now
        _write_tasks()


def start_subagent_task(agent_id: str, label: str):
    """A background/delegated agent (SubagentStart) has begun running
    independently of this turn's own tool-call stream — track it by its
    own agent_id so the panel keeps showing it for its real lifetime.
    Never raises."""
    global _heartbeat_thread
    now = time.time()
    _subagent_tasks[agent_id] = {"id": agent_id, "ts": now, "hb": now,
                                  "label": label, "eta": None}
    if _heartbeat_thread is None or not _heartbeat_thread.is_alive():
        _heartbeat_thread = threading.Thread(target=_heartbeat_loop,
                                              daemon=True)
        _heartbeat_thread.start()
    _write_tasks()


def end_subagent_task(agent_id: str):
    """That background agent (SubagentStop) is done — drop it and
    record it as the newest completion. Never raises."""
    entry = _subagent_tasks.pop(agent_id, None)
    if entry is not None:
        _record_done(entry)
    _write_tasks()


def feed_waveform(pcm: np.ndarray):
    """Feed one PCM block (int16) — throttled, downsampled to 64 points.

    Also re-asserts state="speaking" on the same throttle: this only runs
    while the mouth is audibly playing, so the bus self-heals within
    ~70ms if a stray writer stomps the state mid-speech. (That self-heal
    rule once closed a bug that took a whole evening to find.)"""
    global _last_waveform_write
    if pcm.size == 0:
        return
    now = time.time()
    if now - _last_waveform_write < _WAVEFORM_MIN_INTERVAL:
        return
    _last_waveform_write = now
    try:
        idx = np.linspace(0, pcm.size - 1, 64).astype(int)
        raw = pcm[idx].astype(float)
        with open(_WAVEFORM_FILE, "w") as f:
            f.write(json.dumps({"ts": now, "samples": raw.tolist()}))
        if _BH_WAVE:
            norm = np.clip(np.abs(raw) / 32768.0, 0.0, 1.0)
            with open(_BH_WAVE, "w") as f:
                f.write(json.dumps({"ts": now, "samples": norm.tolist()}))
    except (OSError, ValueError):
        pass
    set_state("speaking")


def feed_input_waveform(pcm: np.ndarray):
    """Feed one PCM block (int16) from the open mic — throttled,
    downsampled to 64 points, same shape as feed_waveform's output file.

    Deliberately does not touch .voice_state: ears.py's callers already
    own that transition, and this is just a loudness feed for whoever's
    listening to it, not a state signal."""
    global _last_input_waveform_write
    if pcm.size == 0:
        return
    now = time.time()
    if now - _last_input_waveform_write < _WAVEFORM_MIN_INTERVAL:
        return
    _last_input_waveform_write = now
    try:
        idx = np.linspace(0, pcm.size - 1, 64).astype(int)
        raw = pcm[idx].astype(float)
        with open(_INPUT_WAVEFORM_FILE, "w") as f:
            f.write(json.dumps({"ts": now, "samples": raw.tolist()}))
    except (OSError, ValueError):
        pass


def direction(items):
    """Stage directions the agent wrote into its reply, published at the
    moment the audio carrying them starts playing.

    Your agent can emit `<<anything>>` inline and backtalk will never speak
    it. What the tag MEANS is deliberately not backtalk's business: it
    publishes the raw strings and something else decides. That is the whole
    reason this is a file and not a plugin API.

    The timing is the point, and it is the one part a watcher cannot do for
    itself: these fire when the sentence becomes AUDIBLE, not when the model
    generated it. A screen cue lands on the spoken word instead of seconds
    early. Never raises."""
    if not items:
        return
    try:
        with open(_DIRECTION_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "directions": list(items)}))
    except OSError:
        pass


def reply_done():
    """One reply has finished speaking and its audio has fully drained.

    Distinct from the state going idle, which also happens in the gaps
    BETWEEN sentences of the same reply. Anything waiting for the agent to
    genuinely stop talking wants this rather than a state flicker. Never
    raises."""
    try:
        with open(_REPLY_DONE_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time()}))
    except OSError:
        pass


_rate_limits: dict = {}


def set_rate_limit(window: str, utilization, resets_at):
    """One usage window's reading — how much of the plan is spent.

    Merged rather than replaced, because the reading arrives one window
    at a time and a face wants to draw both at once. `utilization` is a
    0..1 fraction (or None when the window has not reported a number
    yet, which is a real state and not an error); `resets_at` is a unix
    epoch.

    NOTHING CALLS THIS UNLESS show_usage IS ON. That is a privacy
    default, not a performance one: this is the account holder's own
    spend, and it renders on a face that may well be pointed at a
    camera. It never appears without being asked for. (Community fix,
    ai-visualizer issue #1.)

    Never raises."""
    if not window:
        return
    _rate_limits[window] = {"utilization": utilization,
                            "resets_at": resets_at}
    try:
        with open(_RATE_LIMIT_FILE, "w") as f:
            f.write(json.dumps(_rate_limits))
    except OSError:
        pass


def _player_cmd(path: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["afplay", "-v", "0.35", path]
    for cand in ("ffplay", "aplay", "paplay"):
        from shutil import which
        if which(cand):
            if cand == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel",
                        "quiet", "-volume", "35", path]
            return [cand, path]
    return None


def static_start():
    """Optional thinking sound — plays while the brain works."""
    global _static_proc
    if not _THINKING_SOUND or not os.path.exists(_THINKING_SOUND):
        return
    static_stop()
    cmd = _player_cmd(_THINKING_SOUND)
    if not cmd:
        return
    try:
        _static_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(_LOADING_PID_FILE, "w") as f:
            f.write(str(_static_proc.pid))
    except OSError:
        _static_proc = None


def static_stop():
    global _static_proc
    if _static_proc is not None:
        try:
            _static_proc.terminate()
        except OSError:
            pass
        _static_proc = None
    try:
        os.remove(_LOADING_PID_FILE)
    except OSError:
        pass
