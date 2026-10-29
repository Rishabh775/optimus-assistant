# Soundwave Android app

A thin native Android shell (Capacitor) around the same Soundwave UI the
phone browser uses. The shell exists for the things a PWA cannot do well:

- real app icon + splash, no browser chrome or address bar
- camera / photo picker and document upload (WebView file chooser)
- microphone permission handled natively (voice notes -> `/api/transcribe`)
- keeps the session warm (singleTask activity)

The UI itself is still `jarvis/server/static/index.html` -- one codebase,
so improvements land in both the browser and the app.

## How it connects

`capacitor.config.json` points the WebView at the tailnet HTTPS URL:

```json
"server": { "url": "https://cachyos-x8664.tailb763b1.ts.net" }
```

That means the phone must have Tailscale ON, exactly like the browser flow.
If your tailnet name ever changes, edit that URL, run `npx cap sync android`,
and rebuild.

## Build

Requires JDK 17+ and the Android SDK (platform 35, build-tools 35).

```bash
cd mobile
export ANDROID_HOME=$HOME/Android/Sdk
npm install                 # once
npx cap sync android        # after changing capacitor.config.json
cd android && ./gradlew assembleDebug
# -> android/app/build/outputs/apk/debug/app-debug.apk
```

## Install on the phone

1. Copy `app-debug.apk` to the phone (USB, `python -m http.server`, or
   `adb install app-debug.apk` when USB debugging is on).
2. Open the APK, allow "Install unknown apps" for your file manager/browser.
3. Launch Soundwave, paste the `JARVIS_APP_TOKEN` from the PC's `.env`.

The debug APK is signed with the local debug key -- fine for personal
sideloading. For a Play-Store-style release APK, add a keystore and run
`./gradlew assembleRelease`.
