# Optimus (jarvis)

Local-first personal AI assistant. Hybrid routing between **Ollama** (local) and the **opencode Go** subscription (cloud).

```
mic -> [wake-word] -> faster-whisper STT -> Agent -> [Router] -> Ollama (simple, opt-in)
                                                         \-> longcat-2.5-preview-free (default cloud, free)
                                                         \-> mimo-v2.6-flash (cloud fallback)
                                                         \-> Kimi K2.7 Code (code)
                       -> 23 tools (self-status/memory/triage/shell/file/web/browser/email/calendar/HA)
                       -> response -> edge-tts (Andrew voice) -> speaker
```

## Install

```bash
cd ~/jarvis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install portaudio19-dev ffmpeg        # for sounddevice + audio
playwright install chromium                     # for browser_* tools
python install_wake_models.py            # install bundled wake-word verifiers
cp .env.example .env                     # then fill in keys
```

The wake-word verifier for "hey soundwave" (plus "hey optimus" and "optimus") is bundled in `wake_models/` and installed into the openwakeword package by the step above. To retrain, see `data/openWakeWord/` (scripts `gen_positive.py` + `train_optimus.py`).

## Run

```bash
python -m jarvis --mode text      # interactive text prompt
python -m jarvis --mode voice     # mic + speaker (push-to-talk)
python -m jarvis --mode hotkey    # press JARVIS_HOTKEY to activate
python -m jarvis --mode wake      # "optimus, ..." (chirps on wake, then
                                  #  conversation for follow-ups)
python -m jarvis --query "..."    # one-shot
```

**In wake mode, speak *after* the greeting ends** — the mic is closed while Optimus talks (that's the echo guard), so commands spoken during the greeting are lost. Greetings are now 1-3 words (~1s).

## Tools (23)

| Tool | What |
|---|---|
| `self_status` | report the assistant's own live config/state (introspection) |
| `remember_fact`, `recall_facts`, `forget_fact` | durable user memory (name, preferences) persisted to `data/facts.jsonl` |
| `shell_run`, `read_file`, `write_file` | local system |
| `web_search`, `browser_open`, `browser_extract`, `browser_screenshot` | web |
| `email_triage` | rank emails by importance (VIP senders + keywords) |
| `email_list`, `email_read`, `email_send`, `email_unread_count` | email (IMAP/SMTP) |
| `calendar_list`, `calendar_today`, `calendar_add` | calendar (CalDAV) |
| `ha_list_entities`, `ha_state`, `ha_call`, `ha_search` | Home Assistant REST |

Tool calls have a budget of 12 per turn (`JARVIS_MAX_STEPS`). If the agent runs out it does one final no-tools summary call so you still get whatever it found.

## Routing

| Trigger | Provider | Model |
|---|---|---|
| action verbs (list/read/run/...) | cloud | `longcat-2.5-preview-free` |
| code/refactor/bug | cloud | `kimi-k2.7-code` |
| plan/analyze/long | cloud | `longcat-2.5-preview-free` |
| simple chat | local ollama (opt-in: `JARVIS_LOCAL_OK=1`) | `llama3.2:3b` |

7B local models are ~5 tok/s on a 4-core CPU — 3B is the usable local tier.
Keyword routing uses word boundaries and always routes on the original user prompt (never on mid-loop tool results).

**Cloud resilience:** the primary cloud model can be set to any free preview; if it fails mid-call, the router retries once with `JARVIS_CLOUD_FALLBACK` (default `mimo-v2.6-flash` — cheapest paid tier at $0.14/$0.28 per 1M tokens, 30,100 requests/5h).

## Wake word

The bundled custom verifier fires on **"hey soundwave"** (retrained with adversarial negatives: ordinary-speech false-fire ~0.1%). See "Retraining the wake word" below to build your own.

| Knob | Default | Notes |
|---|---|---|
| `JARVIS_WAKE_WORDS` | `hey_soundwave` | must match a `*.json`+`*.onnx` pair in the openwakeword resources dir |
| `JARVIS_WAKE_THRESHOLD` | `0.7` | high recall (easy to trigger); raise toward 0.85 only if you get false wakes |
| `JARVIS_WAKE_DEBOUNCE` | `1` | **must stay 1**: the sharper model peaks on a single 1.2s chunk (2+ never fires) |
| `JARVIS_WAKE_CHUNK_MS` | `1200` | chunk size fed to the classifier |
| `JARVIS_WAKE_ENERGY_MIN` | `1000` | int16 peak gate before classifier inference |

False-wake protection has three layers regardless of threshold tuning: (1) wake detection will not arm while our own TTS is playing, (2) empty command after a wake triggers a 2s cooldown before re-arming, (3) the echo guard rejects captured speech that overlaps with what Optimus just said. So you can afford a generous threshold. Whisper wake fallback (when no custom model exists) CPU-pegs a core — keep it as a diagnostic only.

### Retraining the wake word

`training/` contains the scripts that produced the bundled models.

Prereqs (inside the venv): `openwakeword`, `torch`, `edge-tts`, `onnxruntime`,
`ffmpeg`. You also need `validation_set_features.npy` — openWakeWord's negative
feature bank (~180 MB). Put it in `training/`.

```bash
cd training
# 1) positives (14 edge-tts voices, pitch/rate variants; also emits bare-word + "hey X" forms)
python gen_positive.py --phrase "hey soundwave" --out-dir out_hey_soundwave/hey_soundwave/positive_train --n 500
python gen_positive.py --phrase "hey soundwave" --out-dir out_hey_soundwave/hey_soundwave/positive_test  --n 120
# 2) adversarial negatives (competing wake words + homophones) -> negative_train / negative_test
#    (run gen_positive.py per phrase, e.g. "hey jarvis", "soundcloud", "wavelength", then move into the dir)
# 3) train (prints recall + false-wake numbers and a suggested threshold)
python train_optimus.py --data-dir out_hey_soundwave/hey_soundwave --model-name hey_soundwave --epochs 25
```

It writes `training/my_custom_model/<name>.onnx` + `.json`. To deploy: copy both
into `wake_models/`, run `python install_wake_models.py`, then set
`JARVIS_WAKE_WORDS=<name>` in `.env`. Use `JARVIS_WAKE_DEBOUNCE=1` and pick
`JARVIS_WAKE_THRESHOLD` from the printed false-wake/recall trade-off
(0.7 is a good start).

## Voice pipeline thresholds

| Setting | Default | What it does |
|---|---|---|
| `JARVIS_STT_END_SILENCE` | `1800`ms | mid-speech pause tolerance; raise if you still get cut off while thinking |
| `JARVIS_STT_MAX_NO_SPEECH` | `0.6` | confidence gate: drop transcript if it also has low logprob |
| `JARVIS_STT_MIN_LOGPROB` | `-1.0` | confidence gate: drop transcript if no_speech_prob is also high |
| `JARVIS_CLOUD_COOLDOWN` | `90`s | after a cloud failure, use the fallback for this long (circuit breaker) |
| `JARVIS_STT_ENERGY_MULT` | `1.0` | peak-based mic gate (matches wake scale); floor 400 — soft speech stays "voiced" |
| `JARVIS_STT_GATE_MAX` | `6000` | gate ceiling; ambient is a median (transient-proof) so a loud moment can't make it deaf |
| `JARVIS_STT_TAIL` | `0.35`s | silence after TTS stops before sampling/calibrating (speaker tail margin) |
| `JARVIS_STT_COMPUTE` | `int8` | faster-whisper compute type |
| `JARVIS_STT_THREADS` | `4` | faster-whisper CPU threads |
| `JARVIS_FOLLOWUP` | `1` | continued conversation window after each answer |
| `JARVIS_FOLLOWUP_SECONDS` | `6` | how long the mic stays open for a follow-up |
| `JARVIS_SOUNDS` | `1` | ack chirp on wake + end tone after speech captured |
| `JARVIS_GREET` | `1` | short spoken greeting on wake |
| `JARVIS_GREETINGS` | `Yes, boss?|Go ahead.|Boss?|Listening.|I'm here.|Shoot.` | `|`-separated custom pool |
| `JARVIS_TTS_VOICE` | `en-US-AndrewMultilingualNeural` | best free neural male |
| `JARVIS_TTS_RATE` | `+0%` | e.g. `+10%` for snappier playback |
| `JARVIS_HOTKEY` | `ctrl+alt+j` | supports `ctrl`, `alt`, `shift`, `cmd/super/win`, + single letter |

`voice.listen` also retries once without VAD if the energy gate saw speech but the Silero VAD stripped the audio — protects against quiet utterances being lost. Transcripts that look like silence/noise (high `no_speech_prob` AND low `avg_logprob`) are dropped instead of sent to the LLM.

**Fully-offline note:** STT is local (faster-whisper) and memory is local, but TTS uses edge-tts (Microsoft, cloud) by default. To go fully offline, install `piper-tts` + `espeak-ng` and a Piper voice, and point `JARVIS_TTS_PLAYER`/engine at it — not enabled by default because `espeak-ng` is a system dependency.

## Configure integrations

### Home Assistant
```bash
# In HA: Profile -> Long-Lived Access Tokens -> Create
JARVIS_HA_URL=http://homeassistant.local:8123
JARVIS_HA_TOKEN=eyJ...
```

### Email (Gmail example)
1. Enable 2FA + IMAP in Gmail settings
2. https://myaccount.google.com/apppasswords -> create app password
3. Set `JARVIS_EMAIL_USER` and `JARVIS_EMAIL_PASS` (the app password)

### Calendar (iCloud example)
```bash
JARVIS_CALDAV_URL=https://caldav.icloud.com
JARVIS_CALDAV_USER=you@icloud.com
JARVIS_CALDAV_PASS=<app-specific-password>
JARVIS_CALDAV_CALENDAR=  # leave blank for default
```

## Files

```
jarvis/                      # python package
├── __main__.py      # entrypoint (text/voice/hotkey/wake/--query)
├── agent.py         # tool-calling loop + system prompt + step-budget summary
├── router.py        # Ollama + opencode hybrid routing + fallback retry
├── builtin_tools.py # tool implementations + schemas + sandbox
├── voice.py         # faster-whisper STT (int8, persistent) + edge-tts TTS
├── wake.py          # openWakeWord hotword (custom + ONNX + whisper fallback)
├── memory.py        # Chroma persistent memory (similarity-gated, pruned)
├── facts.py         # durable user facts (facts.jsonl)
├── config.py        # central settings + validation
├── audio.py         # shared gain/energy/warning helpers
├── log.py           # leveled logging (JARVIS_LOG_LEVEL)
├── sounds.py        # ack/end chirps (generated, no assets)
└── tools/
    ├── browser.py       # Playwright
    ├── email.py         # IMAP/SMTP
    ├── calendar.py      # CalDAV
    └── homeassistant.py # REST API

tests/                       # pytest suite (offline: fake mic/player, no network)
wake_models/                 # pre-trained wake-word verifiers (onnx + json)
training/                    # wake-word training scripts (gen_positive, train_optimus)
install_wake_models.py       # copies wake_models/ into openwakeword
setup_email_pass.py          # helper: store email app-password in .env
requirements.txt
requirements-dev.txt
pytest.ini
.env.example
LICENSE                      # MIT
```

## Security

- `.env` is chmod 600 and gitignored
- `shell_run` executes in `~/jarvis/workspace/` with a **scrubbed environment** (keys/tokens/passwords removed from child processes) and a destructive-command blocklist (rm -r outside workspace, mkfs, dd to /dev, fork bombs, `curl|sh`, shutdown/reboot, etc.)
- `read_file` limited to `$HOME` + `/tmp`; secret files (`.env`, keys, tokens, credentials) blocked
- `write_file` limited to `workspace/` by default (`JARVIS_ALLOW_ANY_PATH=1` lifts to `$HOME`)
- **Confirmation gate** (`JARVIS_CONFIRM=1`): the agent asks before consequential actions — risky shell commands (rm/mv/sudo/redirects/installs), sending email, overwriting files. In voice mode it asks aloud and waits for "yes"; in text mode it prompts. This is the main defense against the email/web → shell prompt-injection chain.
- The blocklist is **not** a sandbox: content read via email/browser flows into the model's context, so a crafted email/web page can still try to steer `shell_run`. For defense in depth, run as a dedicated user or wrap shell in `firejail`/`bubblewrap`.

## Tested examples

```bash
python -m jarvis --query "list files in current dir"
python -m jarvis --query "what's on my calendar"
python -m jarvis --query "turn off living room light"
python -m jarvis --query "how many unread emails"
python -m jarvis --query "open github.com and tell me what's trending"
```
## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest          # offline; no mic, no API key needed
```

## Logging

Diagnostics go through `jarvis/log.py`. Default output is unchanged; raise verbosity or add timestamps via env:

```bash
JARVIS_LOG_LEVEL=DEBUG
JARVIS_LOG_FORMAT="%(asctime)s %(levelname)s %(name)s: %(message)s"
```

## License

MIT - see [LICENSE](LICENSE).
