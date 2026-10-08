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
"""The hard path boundary — config.blocked_paths, enforced as a real wall.

This is deliberately NOT part of main.py's spoken permission gate. That
gate is a can_use_tool callback, and the SDK skips can_use_tool entirely
under permission_mode "bypassPermissions" (see the CanUseToolShadowedWarning
the SDK itself raises for exactly this reason) -- so anything built only
into that gate would stop protecting the moment someone flips to bypass
mode, which defeats the point for a path that's supposed to be off-limits
"under any circumstance", not "off-limits unless you're in a hurry".

A PreToolUse hook is different: the SDK fires it for every tool call no
matter what permission_mode is active, bypassPermissions included -- it's
the mechanism the SDK's own docs point to when you need a check that can't
be shadowed. That's what makes blocked_paths an actual technical wall
instead of a setting someone could switch past.

Checked against every tool, not just file tools: Bash can read a blocked
file with `cat` as easily as Read can open it directly, so the guard scans
the raw command string too, not just file_path-shaped fields.
"""
import os

from backtalk.config import CFG
from backtalk.vlog import log


def _candidate_paths(tool_name: str, tool_input: dict) -> list[str]:
    """Every string in this tool call worth checking against blocked_paths."""
    d = tool_input or {}
    out = []
    for key in ("file_path", "path", "notebook_path"):
        v = d.get(key)
        if v:
            out.append(str(v))
    if tool_name == "Bash":
        cmd = d.get("command")
        if cmd:
            out.append(str(cmd))
    return out


async def guard_blocked_paths(input_data, tool_use_id, context):
    """PreToolUse hook: deny outright if this call touches a blocked path.

    Substring match against the expanded path, deliberately simple and
    conservative -- a false-positive block (e.g. a blocked folder name
    that also appears harmlessly in an unrelated path) just means an
    occasional over-cautious denial, which is the safe direction to be
    wrong in. A false NEGATIVE here is the one this exists to prevent.
    """
    blocked = CFG.get("blocked_paths") or []
    if not blocked:
        return {}

    tool_name = input_data.get("tool_name", "")
    candidates = _candidate_paths(tool_name, input_data.get("tool_input"))
    if not candidates:
        return {}

    for raw in candidates:
        norm = os.path.expanduser(raw)
        for b in blocked:
            if b and b in norm:
                reason = (f"{tool_name} touches a path this install is "
                          f"configured to never access ({b}).")
                log(f"[path_guard] DENIED: {reason}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
    return {}
