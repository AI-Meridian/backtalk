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
"""Pre-render the filler_sound asset once, offline, through whichever TTS
backend backtalk.json already has configured (ElevenLabs if enabled, Kokoro
otherwise) — same voice the agent actually speaks with live. Run this once
after the voice is chosen (by hand, or as an install step) so static_start()
has zero-latency audio to play; filler lines are never synthesized live.

    python -m backtalk.gen_filler "Still working on that, sir." assets/filler_ack.wav
"""
import sys
import wave

import numpy as np


def generate(text: str, out_path: str) -> str:
    from backtalk import mouth
    chunks = []
    rate = None
    for sr, pcm in mouth.synth_stream(text):
        rate = sr
        chunks.append(pcm)
    if not chunks:
        raise RuntimeError("TTS returned no audio for the filler line")
    audio = np.concatenate(chunks)
    with wave.open(out_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio.tobytes())
    return out_path


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    path = generate(sys.argv[1], sys.argv[2])
    print(f"[gen_filler] wrote {path}")


if __name__ == "__main__":
    main()
