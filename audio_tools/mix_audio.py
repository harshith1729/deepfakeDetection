import librosa
import numpy as np
import soundfile as sf
import argparse
import os

# -------------------------------
# ARGUMENT PARSING
# -------------------------------
parser = argparse.ArgumentParser(
    description="Mix (overlap) a voice audio and a noise audio"
)

parser.add_argument(
    "voice",
    type=str,
    help="Path to clean voice audio (.wav)"
)

parser.add_argument(
    "noise",
    type=str,
    help="Path to noise/disturbance audio (.wav)"
)

parser.add_argument(
    "--noise_level",
    type=float,
    default=0.3,
    help="Noise volume multiplier (default=0.3)"
)

args = parser.parse_args()

VOICE_PATH = args.voice
NOISE_PATH = args.noise
NOISE_LEVEL = args.noise_level

OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "voice_with_noise.wav")

# -------------------------------
# LOAD AUDIO FILES
# -------------------------------
print("🔊 Loading voice:", VOICE_PATH)
voice, sr = librosa.load(VOICE_PATH, sr=16000)

print("🔊 Loading noise:", NOISE_PATH)
noise, _ = librosa.load(NOISE_PATH, sr=16000)

# -------------------------------
# MATCH LENGTH
# -------------------------------
min_len = min(len(voice), len(noise))
voice = voice[:min_len]
noise = noise[:min_len]

# -------------------------------
# MIX AUDIO
# -------------------------------
noise = noise * NOISE_LEVEL
mixed = voice + noise

# Normalize to avoid clipping
max_val = np.max(np.abs(mixed))
if max_val > 0:
    mixed = mixed / max_val

# -------------------------------
# SAVE OUTPUT
# -------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
sf.write(OUTPUT_PATH, mixed, sr)

print("✅ Mixed audio saved at:", OUTPUT_PATH)
print(f"🎚 Noise level used: {NOISE_LEVEL}")
