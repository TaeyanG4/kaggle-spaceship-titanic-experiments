"""Audible alert for a confirmed improvement (promotion) only; never for routine completions."""

from __future__ import annotations

import sys

SOUND = r"C:\Windows\Media\Windows Notify System Generic.wav"


def improvement_alert(message: str) -> None:
    print(f"IMPROVEMENT: {message}", flush=True)
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(SOUND, winsound.SND_FILENAME)


if __name__ == "__main__":
    improvement_alert(" ".join(sys.argv[1:]) or "confirmed improvement")
