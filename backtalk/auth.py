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
"""Is Claude Code signed in? The one question the voice line could not
answer about itself, and the field case that made it matter.

2026-10-19: the account had been signed out. The voice line booted,
connected the SDK (that part works signed-out), sent its warmup ping,
got back an error result carrying zero sentences, counted "the stream
ended" as success, logged "brain warm", and then answered nothing for
twenty minutes. Face on idle, no reply, no spoken reason. Three layers
blind to one fault.

This module is the preflight and the vocabulary for that fault:

  status()          run `claude auth status --json`, return a plain dict
  is_auth_error(s)  does an error string from the CLI mean "signed out"
                    (the same wording the CLI itself matches on)

Pure where it can be; the subprocess is confined to status().
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Mirrors the CLI's own classifier for a sign-in problem, so the voice
# line agrees with the terminal about what counts as "signed out".
AUTH_ERROR_RE = re.compile(
    r"not logged in|run /login|please sign in|authentication (?:failed|error)"
    r"|invalid api key|oauth token|token (?:has )?expired|token (?:was )?revoked",
    re.IGNORECASE)

SPOKEN_SIGNED_OUT = (
    "Heads up. I'm not signed in to Claude, so I can't think right now. "
    "Hit the sign in button on my face, or run claude auth login in a "
    "terminal, and I'll pick up the moment it lands.")
SPOKEN_NO_CLI = (
    "I can't find Claude Code on this machine, so I have no brain to "
    "connect to. Install it, sign in, and I'll pick up from there.")
SPOKEN_STILL_WAITING = "Still waiting on the Claude sign-in."
SPOKEN_SIGNED_IN = "Signed in. Connecting my brain now."


def is_auth_error(text) -> bool:
    return bool(text) and bool(AUTH_ERROR_RE.search(str(text)))


def find_claude() -> str | None:
    """The Claude Code binary: PATH first, then the usual homes. A
    launcher started from the Dock or an app bundle inherits a bare
    PATH without ~/.local/bin, which is where the installer puts it."""
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "claude",
        home / ".claude" / "local" / "claude",
        Path("/opt/homebrew/bin/claude"),
        Path("/usr/local/bin/claude"),
    ]
    if sys.platform == "win32":
        candidates.append(home / ".local" / "bin" / "claude.exe")
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            candidates.append(Path(appdata) / "npm" / "claude.cmd")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def parse_status(stdout: str, returncode: int) -> dict:
    """`claude auth status --json` output -> {logged_in, email, plan,
    method, error}. Tolerant: the shape belongs to the CLI, and an
    unreadable answer must read as "signed out, and here's why", never
    crash the boot."""
    out = {"logged_in": False, "email": "", "plan": "", "method": "",
           "error": ""}
    try:
        data = json.loads(stdout or "{}")
    except ValueError:
        out["error"] = (f"unreadable auth status (exit {returncode}): "
                        f"{(stdout or '').strip()[:120]}")
        return out
    if not isinstance(data, dict):
        out["error"] = "unexpected auth status shape"
        return out
    out["logged_in"] = bool(data.get("loggedIn"))
    out["email"] = str(data.get("email") or "")
    out["plan"] = str(data.get("subscriptionType") or "")
    out["method"] = str(data.get("authMethod") or "")
    if not out["logged_in"] and returncode not in (0, 1):
        out["error"] = f"auth status exited {returncode}"
    return out


def status() -> dict:
    """One real check, bounded, never raises. Adds cli_found to the
    parse_status fields."""
    cli = find_claude()
    if not cli:
        return {"cli_found": False, "logged_in": False, "email": "",
                "plan": "", "method": "",
                "error": "claude not found on this machine"}
    try:
        r = subprocess.run([cli, "auth", "status", "--json"],
                           capture_output=True, text=True, timeout=15)
        out = parse_status(r.stdout, r.returncode)
    except subprocess.TimeoutExpired:
        out = {"logged_in": False, "email": "", "plan": "", "method": "",
               "error": "auth status timed out"}
    except OSError as e:
        out = {"logged_in": False, "email": "", "plan": "", "method": "",
               "error": f"could not run claude: {e}"}
    out["cli_found"] = True
    return out
