"""Shared low-level audio helpers.

Both voice.py (capture/STT) and wake.py (hotword) previously duplicated gain
and energy-gate logic -- and they drifted once (RMS vs peak). Single source now.
"""
import os
import warnings

import numpy as np

SAMPLE_RATE = 16000


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """Software gain for quiet mics; linear amplify + clip to int16 range."""
    if gain == 1.0:
        return audio
    boosted = audio.astype(np.int32) * gain
    return np.clip(boosted, -32768, 32767).astype(np.int16)


def mic_gain() -> float:
    return float(os.getenv("JARVIS_MIC_GAIN", "1.0"))


def peak(audio: np.ndarray, gain: float = None) -> float:
    """Peak absolute amplitude after applying mic gain."""
    if gain is None:
        gain = mic_gain()
    return float(np.abs(apply_gain(audio, gain)).max())


def energy_gate(audio: np.ndarray, threshold: int = 1000) -> bool:
    """True if audio is loud enough to be speech (used before inference)."""
    return peak(audio) > threshold


def silence_warnings() -> None:
    """Re-apply noise filters. openwakeword/onnxruntime reset Python's warning
    filters at import, resurrecting sounddevice's NumPy spam + ResourceWarnings."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
