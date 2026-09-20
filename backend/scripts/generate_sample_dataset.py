import os
import wave
import struct
import math
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"

CLASSES = ["normal", "misfire", "bearing_fault", "valve_fault", "knocking"]
SAMPLE_RATE = 22050
DURATION = 4.0  # seconds

def generate_engine_audio(fault_class: str, filename: str):
    num_samples = int(SAMPLE_RATE * DURATION)
    audio_data = []

    # Engine fundamental RPM frequency (approx 30 Hz idle = 1800 RPM)
    f0 = 35.0

    for i in range(num_samples):
        t = i / SAMPLE_RATE

        # Base engine combustion rumble (harmonics)
        signal = 0.4 * math.sin(2 * math.pi * f0 * t) + \
                 0.25 * math.sin(2 * math.pi * (2 * f0) * t) + \
                 0.15 * math.sin(2 * math.pi * (4 * f0) * t) + \
                 0.05 * (random.random() * 2 - 1)  # smooth background noise

        # Class specific acoustic modifications
        if fault_class == "misfire":
            # Periodic drop in engine output + sudden transient click every 0.6 seconds
            cycle = t % 0.6
            if cycle < 0.08:
                signal *= 0.2  # drop power
                signal += 0.3 * (random.random() * 2 - 1)  # misfire pop

        elif fault_class == "bearing_fault":
            # High frequency friction screech / grinding noise
            friction_freq = 2800.0
            signal += 0.35 * math.sin(2 * math.pi * friction_freq * t) * (1 + 0.3 * math.sin(2 * math.pi * 5 * t))
            signal += 0.15 * (random.random() * 2 - 1)

        elif fault_class == "valve_fault":
            # Rapid metallic tapping (at half engine RPM = 17.5 Hz)
            tap_cycle = t % (1 / 17.5)
            if tap_cycle < 0.015:
                signal += 0.6 * math.sin(2 * math.pi * 4200.0 * t) * math.exp(-tap_cycle * 200)

        elif fault_class == "knocking":
            # High energy sudden metallic impact knock every 0.4 seconds
            knock_cycle = t % 0.4
            if knock_cycle < 0.02:
                signal += 0.8 * math.sin(2 * math.pi * 5500.0 * t) * math.exp(-knock_cycle * 300)

        # Normalize and clip
        signal = max(-1.0, min(1.0, signal))
        packed_sample = struct.pack('<h', int(signal * 32767.0))
        audio_data.append(packed_sample)

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b''.join(audio_data))

def main():
    print("Generating synthetic engine sound dataset...")
    total_generated = 0
    
    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 6 audio samples per class (total 30 samples for fast demo training & evaluation)
        for idx in range(1, 7):
            filepath = cls_dir / f"engine_{cls}_sample_{idx}.wav"
            generate_engine_audio(cls, str(filepath))
            total_generated += 1

    print(f"Generated {total_generated} engine audio samples across {len(CLASSES)} classes in {DATASET_DIR}")

if __name__ == "__main__":
    main()
