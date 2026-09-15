"""
Voice Sample Recorder
Run this once to capture your voice clone reference.
Speak naturally for 15-30 seconds when prompted.
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLE_RATE = 22050
DURATION    = 20
OUTPUT_DIR  = Path("data/voices")
OUTPUT_FILE = OUTPUT_DIR / "my_voice.wav"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("  Robot Agent — Voice Clone Sample Recorder")
print("=" * 55)
print()
print("Tips for best cloning quality:")
print("  • Quiet room, no background noise")
print("  • Speak clearly at natural pace and volume")
print("  • Read a paragraph of text aloud works great")
print("  • Keep consistent distance from microphone")
print()
print(f"Recording duration: {DURATION} seconds")
print()
print("Press Enter when ready to record →", end=" ", flush=True)
input()

print()
print("🔴 Recording NOW — speak naturally...")
print(f"   (recording for {DURATION} seconds)")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32'
)
sd.wait()

# Normalise amplitude
peak = np.max(np.abs(recording))
if peak > 0:
    recording = recording / peak * 0.95

sf.write(str(OUTPUT_FILE), recording, SAMPLE_RATE)

size_kb = OUTPUT_FILE.stat().st_size // 1024
print()
print(f"✅ Saved: {OUTPUT_FILE}  ({size_kb} KB)")
print()
print("Next steps:")
print("  1. Open Settings → Voice Lab in Robot Agent")
print(f"  2. Upload the file: {OUTPUT_FILE}")
print("  3. Set TTS Provider to 'coqui'")
print("  4. Click Save & Apply")
