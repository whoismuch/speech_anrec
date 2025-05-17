import numpy as np
import soundfile as sf
import torch
import torchaudio

def overlaps_with_existing(start, end, existing_intervals):
    for existing_start, existing_end in existing_intervals:
        if not (end <= existing_start or start >= existing_end):
            return True
    return False

def combine_segments(mono_segments, target_segments, target_speaker, y, sr, output_path):
    print("🔜 Объединяем все фрагменты целевого спикера...")

    all_chunks = []
    used_intervals = []

    # --- 1. Mono сегменты
    for start, end, speaker in mono_segments:
        if speaker == target_speaker:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk = y[start_sample:end_sample]
            all_chunks.append((start, chunk))
            used_intervals.append((start, end))

    # --- 2. Разделённые сегменты (без перекрытия)
    resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sr)

    for path, start, end in target_segments:
        if overlaps_with_existing(start, end, used_intervals):
            continue  # пропускаем дубликат

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
        used_intervals.append((start, end))

    # --- Сортировка и объединение
    all_chunks.sort(key=lambda x: x[0])

    if all_chunks:
        combined_audio = np.concatenate([chunk for _, chunk in all_chunks])
    else:
        print("⚠️ Нет подходящих фрагментов — создаётся пустой WAV.")
        combined_audio = np.zeros(1, dtype=np.float32)  # 1 семпл тишины для валидного WAV

    # --- Сохраняем
    sf.write(output_path, combined_audio, sr)
    print(f"✅ Финальный WAV сохранён: {output_path}")
