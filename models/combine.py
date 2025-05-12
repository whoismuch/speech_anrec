# models/combine.py

import numpy as np
import soundfile as sf
import torch
import torchaudio

def combine_segments(mono_segments, target_segments, target_speaker, y, sr, output_path):
    print("🔜 Объединяем все фрагменты целевого спикера...")

    mono_target_segments = [
        (start, end) for start, end, speaker in mono_segments
        if speaker == target_speaker
    ]

    collected_chunks = []

    # --- 1. Mono сегменты из исходного аудио
    for start, end in mono_target_segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        collected_chunks.append(chunk)

    # --- 2. Разделённые сегменты (8000 → 16000 Гц)
    resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sr)

    for path, start, end in target_segments:
        chunk, chunk_sr = sf.read(path)

        # Ресемплируем при необходимости
        if chunk_sr != sr:
            if chunk.ndim == 1:
                chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            else:
                chunk = torch.tensor(chunk.T, dtype=torch.float32)

            chunk = resampler(chunk).numpy()
            chunk = chunk.mean(axis=0) if chunk.ndim > 1 else chunk  # моно

        elif chunk.ndim == 2:
            chunk = chunk.mean(axis=1)

        collected_chunks.append(chunk)

    # --- Объединяем
    combined_audio = np.concatenate(collected_chunks)

    # --- Сохраняем
    sf.write(output_path, combined_audio, sr)
    print(f"✅ Финальный WAV сохранён: {output_path}")
