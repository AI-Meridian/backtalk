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
"""THE INBOUND HALF OF THE BUS — the one direction signals.py never goes.

Every other file in signals.py is backtalk talking outward to a face.
This is the reverse: a face (ai-visualizer's server, via its /message
and /control endpoints) drops ONE JSON file per command into
<signals_dir>/.voice_inbox/, and this module is what claims them.

Claim-by-delete: poll_once() parses a file and removes it in the same
pass, so a command can never be processed twice even if two things
read the directory around the same moment. A malformed file is logged
and removed, never retried — a bad file sitting there forever, retried
forever, helps nobody.

Command shapes, one per file:

    {"type": "message", "text": "..."}        -> fed to the exact same
                                                   handle(text) a typed
                                                   or spoken utterance
                                                   uses. An attachment
                                                   is never a separate
                                                   field: its absolute
                                                   path is already part
                                                   of "text", the same
                                                   way a path pasted
                                                   into a terminal is
                                                   just more text.
    {"type": "model",  "value": "fast"|"deep"|"fable"|"quick"}
    {"type": "effort", "value": "low"|"medium"|"high"|"xhigh"|"max"}

MODEL_TIERS and EFFORTS are the single source of truth for valid
values — main.py imports EFFORTS rather than keeping its own copy, and
ai-visualizer's server.py keeps its own literal copy for a fast HTTP
400 (the two processes share no package), re-validated here regardless
so nothing is ever trusted just because of which directory it came
from.
"""
import json
import os
import queue
import threading
import time

from backtalk.config import CFG
from backtalk.vlog import log

INBOX_DIRNAME = ".voice_inbox"
MODEL_TIERS = {"fast": "model", "deep": "deep_model", "fable": "fable_model",
               "quick": "haiku_model"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def inbox_dir() -> str:
    return os.path.join(CFG["signals_dir"], INBOX_DIRNAME)


def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True, mode=0o700)
    except OSError:
        pass


def poll_once() -> list[dict]:
    """Scan inbox_dir(), parse+delete each *.json file, return valid
    command dicts in filename order (filenames are a monotonic-ns
    prefix, so lexicographic order is chronological order). Malformed
    or unreadable files are logged and removed, never retried, never
    raised. Returns [] if the directory doesn't exist yet."""
    d = inbox_dir()
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        return []
    out = []
    for name in names:
        path = os.path.join(d, name)
        try:
            with open(path) as f:
                cmd = json.load(f)
        except (OSError, ValueError) as e:
            log(f"[inbox] dropping unreadable command file {name}: {e}")
            cmd = None
        try:
            os.remove(path)
        except OSError:
            pass
        if isinstance(cmd, dict):
            out.append(cmd)
        elif cmd is not None:
            log(f"[inbox] dropping non-object command file {name}")
    return out


def start_watcher(q: "queue.Queue[dict]", interval: float = 0.15) -> threading.Thread:
    """Daemon thread: poll_once() -> queue, sleep, repeat. Drained by
    main.py's asyncio select loop via run_in_executor(None, q.get),
    the exact pattern already used for typed stdin input."""
    _ensure_dir(inbox_dir())

    def _loop():
        while True:
            for cmd in poll_once():
                q.put(cmd)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


_UPLOAD_MAX_AGE_S = 7 * 24 * 3600   # retention: see config.py's uploads_dir


def sweep_uploads():
    """Delete anything under uploads_dir older than 7 days. Called
    from the same watcher loop's thread via start_upload_sweeper, not
    the asyncio loop — this is pure filesystem housekeeping, never
    anything Claude or a face is waiting on. Never raises."""
    d = CFG.get("uploads_dir") or ""
    if not d:
        return
    now = time.time()
    try:
        names = os.listdir(d)
    except OSError:
        return
    for name in names:
        path = os.path.join(d, name)
        try:
            if now - os.stat(path).st_mtime > _UPLOAD_MAX_AGE_S:
                os.remove(path)
        except OSError:
            pass


def start_upload_sweeper(interval: float = 3600.0) -> threading.Thread:
    """Daemon thread, independent of the command watcher: a quiet
    hourly sweep so a long-running client install never accumulates
    uploads forever."""
    _ensure_dir(CFG.get("uploads_dir") or "")

    def _loop():
        while True:
            sweep_uploads()
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t
