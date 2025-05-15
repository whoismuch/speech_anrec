import numpy as np
import soundfile as sf
import torch
import torchaudio

def combine_segments(mono_segments, target_segments, target_speaker, y, sr, output_path):
    print("🔜 Объединяем все фрагменты целевого спикера...")

    all_chunks = []

    # --- 1. Mono сегменты
    for start, end, speaker in mono_segments:
        if speaker == target_speaker:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk = y[start_sample:end_sample]
            all_chunks.append((start, chunk))

    # --- 2. Разделённые сегменты
    resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sr)

    for path, start, end in target_segments:
        chunk, chunk_sr = sf.read(path)

        if chunk_sr != sr:
            if chunk.ndim == 1:
                chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            else:
                chunk = torch.tensor(chunk.T, dtype=torch.float32)

            chunk = resampler(chunk).numpy()
            chunk = chunk.mean(axis=0) if chunk.ndim > 1 else chunk  # моно

        elif chunk.ndim == 2:
            chunk = chunk.mean(axis=1)

        all_chunks.append((start, chunk))

    # --- Сортировка по времени начала
    all_chunks.sort(key=lambda x: x[0])

    # --- Склеиваем
    combined_audio = np.concatenate([chunk for _, chunk in all_chunks])

    # --- Сохраняем
    sf.write(output_path, combined_audio, sr)
    print(f"✅ Финальный WAV сохранён: {output_path}")
