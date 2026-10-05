#!/usr/bin/env python
"""Install the bundled wake-word models into the openwakeword package.

Run once after `pip install -r requirements.txt`:

    python install_wake_models.py

Copies wake_models/*.onnx + *.json into
<site-packages>/openwakeword/resources/models/, where jarvis/wake.py looks
for custom verifiers (JARVIS_WAKE_WORDS).
"""
from pathlib import Path

try:
    import openwakeword
except ImportError:
    raise SystemExit("openwakeword not installed - run: pip install -r requirements.txt")

src = Path(__file__).resolve().parent / "wake_models"
dst = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
dst.mkdir(parents=True, exist_ok=True)

if not src.exists():
    raise SystemExit(f"no wake_models/ directory next to {__file__}")

n = 0
for f in sorted(src.iterdir()):
    if f.suffix in (".onnx", ".json"):
        (dst / f.name).write_bytes(f.read_bytes())
        n += 1
print(f"installed {n} model file(s) into {dst}")
