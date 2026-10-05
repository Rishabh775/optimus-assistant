#!/usr/bin/env python
"""One-time helper: store your mail.com app-specific password in ~/jarvis/.env.

The password is typed with no echo and never printed. Safe to delete after use.

    ~/jarvis/.venv/bin/python ~/jarvis/setup_email_pass.py
"""
import getpass
import os
import stat
from pathlib import Path

ENV = Path(__file__).resolve().parent / ".env"


def main():
    if not ENV.exists():
        print(f"no .env found at {ENV}")
        return 1
    pw = getpass.getpass("Paste your mail.com app-specific password (hidden): ").strip()
    if not pw:
        print("empty - nothing changed")
        return 1
    lines = ENV.read_text().splitlines(keepends=True)
    out, found = [], False
    for l in lines:
        if l.startswith("JARVIS_EMAIL_PASS="):
            out.append(f"JARVIS_EMAIL_PASS={pw}\n")
            found = True
        else:
            out.append(l)
    if not found:
        out.append(f"JARVIS_EMAIL_PASS={pw}\n")
    ENV.write_text("".join(out))
    os.chmod(ENV, stat.S_IRUSR | stat.S_IWUSR)  # 600
    print("saved JARVIS_EMAIL_PASS to .env (chmod 600). Nothing was printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
