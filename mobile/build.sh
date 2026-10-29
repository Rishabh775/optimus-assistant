#!/usr/bin/env bash
# Build the Soundwave Android APK.
# Needs: Node 20+, JDK 21 (Capacitor 7), Android SDK platform 35 + build-tools 35.
set -euo pipefail
cd "$(dirname "$0")"

export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
# Capacitor 7 compiles with source/target 21; fall back to a local Temurin 21
if [ -z "${JAVA_HOME:-}" ] && [ -x "$HOME/Android/jdk21/bin/java" ]; then
  export JAVA_HOME="$HOME/Android/jdk21"
fi
export PATH="${JAVA_HOME:+$JAVA_HOME/bin:}$PATH"

[ -d node_modules ] || npm install --no-audit --no-fund
npx cap sync android
( cd android && ./gradlew assembleDebug )

APK="android/app/build/outputs/apk/debug/app-debug.apk"
echo
echo "APK ready: $(pwd)/$APK"
echo "Sideload: adb install -r \"$(pwd)/$APK\"   (or copy it to the phone)"
