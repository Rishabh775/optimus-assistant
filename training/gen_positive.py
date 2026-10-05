"""Generate positive wake-word samples using edge-tts (Microsoft voices).
Outputs 16kHz mono 16-bit PCM WAV files named like 0.wav, 1.wav, ...

Replaces piper-sample-generator since espeak isn't installed.
"""
import argparse
import asyncio
import os
import random
import subprocess
import tempfile
from pathlib import Path

import edge_tts


VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-EricNeural",
    "en-US-DavisNeural", "en-US-AriaNeural", "en-US-TonyNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
]


# Various ways to say the wake phrase - generate slight variations
def _variations(phrase: str) -> list[str]:
    p = phrase.lower()
    out = [p, p.rstrip("."), p + ",", f"hey {p}", f"{p}", f"hey, {p}",
           f"{p.replace('hey ', '')}", f"hey {p.replace('hey ', '')}"]
    # Add filler
    fillers = ["um, " + p, p + " please", "ok " + p, "okay " + p,
               "yo " + p, "so " + p]
    out.extend(fillers)
    return list(dict.fromkeys(out))


async def _tts(text: str, voice: str, out_mp3: str):
    pct = random.choice(["-10%", "-5%", "+0%", "+5%", "+10%"])
    semitone = random.choice(["-2Hz", "-1Hz", "+0Hz", "+1Hz", "+2Hz"])
    comm = edge_tts.Communicate(text, voice, rate=pct, pitch=semitone)
    await comm.save(out_mp3)


def _to_wav(mp3_path: str, wav_path: str):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp3_path, "-ar", "16000", "-ac", "1",
        "-acodec", "pcm_s16le", wav_path
    ], check=True)


async def _gen_one(phrase: str, voice: str, out_path: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_mp3 = f.name
    try:
        await _tts(phrase, voice, tmp_mp3)
        _to_wav(tmp_mp3, out_path)
    finally:
        if os.path.exists(tmp_mp3):
            os.unlink(tmp_mp3)


async def _run(phrase: str, out_dir: str, n: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    variations = _variations(phrase)
    print(f"Generating {n} '{phrase}' samples into {out_dir}")
    sem = asyncio.Semaphore(8)
    counter = {"done": 0, "fail": 0}

    async def task(i: int, retries: int = 3):
        async with sem:
            out_path = os.path.join(out_dir, f"{i}.wav")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                counter["done"] += 1
                return
            for attempt in range(retries):
                text = random.choice(variations)
                voice = random.choice(VOICES)
                try:
                    await _gen_one(text, voice, out_path)
                    counter["done"] += 1
                    break
                except Exception:
                    await asyncio.sleep(0.5)
            else:
                counter["fail"] += 1
            if (counter["done"] + counter["fail"]) % 100 == 0:
                print(f"  done={counter['done']} fail={counter['fail']}")

    await asyncio.gather(*[task(i) for i in range(n)])
    print(f"Final: done={counter['done']} fail={counter['fail']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phrase", default="hey optimus")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    asyncio.run(_run(args.phrase, args.out_dir, args.n))


if __name__ == "__main__":
    main()