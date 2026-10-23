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
"""Jev (TypeSafe) SHADOW-MODE classifier. 2026-10-21.

Classifies every utterance into a 3-tier model (tier1 deterministic /
tier2 light reasoning / tier3 real work) and logs the guess next to the
real turn. Never affects what the agent actually does — classify_shadow
is fire-and-forget and swallows every error, because a Jev outage or a
slow response must be invisible to the voice loop. The point is to
collect real accuracy data across real usage before ever trusting this
to skip the model for real. See "Hugo - Model Routing for Sub-Tasks" in
the vault for the policy this feeds.

Seed the key once:
  security add-generic-password -a "$USER" -s backtalk-typesafe -T /usr/bin/security -w
(prompts for the secret so it never lands in shell history)."""
import subprocess
import time

import httpx

from backtalk.vlog import log

_KEY_SLOT = "backtalk-typesafe"
_key_cache: str | None = None

_QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "Which tier of AI assistant request is this",
        "criteria": {
            "tier1": "A known, deterministic command with one obvious "
                     "action and no reasoning needed (e.g. clear "
                     "session, check status)",
            "tier2": "A light, non-complex question needing a short, "
                     "simple reasoning pass, not real work",
            "tier3": "Real work: coding, diagnosis, multi-step "
                     "reasoning, or anything needing genuine judgment",
        },
    },
}


def _get_key() -> str:
    global _key_cache
    if _key_cache is not None:
        return _key_cache
    key = ""
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", _KEY_SLOT, "-w"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            key = r.stdout.strip()
    except Exception:
        pass
    _key_cache = key
    return key


async def classify_shadow(text: str) -> None:
    key = _get_key()
    if not key:
        log("[jev-shadow] no key in backtalk-typesafe, skipping")
        return
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"state": text, "model": "jev-latest",
                      "questions": _QUESTIONS})
            r.raise_for_status()
        ms = (time.monotonic() - t0) * 1000
        ans = r.json()["answers"]["tier"]
        log(f"[jev-shadow] tier={ans['choice']} "
            f"confidence={ans['confidence']:.2f} latency={ms:.0f}ms "
            f"text={text[:80]!r}")
    except Exception as e:
        log(f"[jev-shadow] failed: {e!r}")
