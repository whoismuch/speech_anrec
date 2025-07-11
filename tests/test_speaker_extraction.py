# tests/test_speaker_extraction.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import soundfile as sf
from scipy.spatial.distance import cosine
import numpy as np
import os

from models.diarization import run_diarization
from models.speaker_extraction import extract_target_speaker
from models.speaker_id import extract_embedding, get_encoder
import librosa
import librosa.display
import argparse


# REFERENCE_PATH = "tests/data/audio/reference/ali_imba_drink.wav"
# AUDIO_DIR = Path("tests/data/audio/ali/test3")
# OUTPUT_DIR = Path("tests/data/output/speaker_extraction_results_FINAL_FINAL_THESIS_THESIS/ali")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

encoder = get_encoder()

def plot_waveform_comparison(original_path, extracted_path, output_path):
    y_orig, sr_orig = sf.read(original_path)
    y_ext, sr_ext = sf.read(extracted_path)

    t_orig = np.linspace(0, len(y_orig) / sr_orig, len(y_orig))
    t_ext = np.linspace(0, len(y_ext) / sr_ext, len(y_ext))

    plt.figure(figsize=(12, 4))
    plt.plot(t_orig, y_orig, label="Многоспикерная запись", alpha=0.5)
    plt.plot(t_ext, y_ext, label="Извлечённый целевой спикер", alpha=0.8)
    plt.title("Сравнение формы сигнала: до и после извлечения целевого говорящего")
    plt.xlabel("Время, сек")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_speaker_segments(mono_segments, multi_segments, target_speaker, output_path):
    fig, ax = plt.subplots(figsize=(12, 2))

    for start, end, speaker in mono_segments:
        if speaker == target_speaker:
            ax.add_patch(Rectangle((start, 0), end - start, 0.9, color='green', label='Mono', alpha=0.6))

    for start, end, speakers in multi_segments:
        if target_speaker in speakers:
            ax.add_patch(Rectangle((start, 1.2), end - start, 0.9, color='orange', label='Overlap', alpha=0.6))

    ax.set_yticks([0, 1.2])
    ax.set_yticklabels(['Mono', 'Overlap'])
    ax.set_xlabel("Время, сек")
    ax.set_title(f"Сегменты речи целевого спикера ({target_speaker})")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_melspectrogram(audio_path, output_path, title):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_segment_list(mono_segments, multi_segments, target_speaker, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Тип\tНачало\tКонец\n")
        for start, end, speaker in mono_segments:
            if speaker == target_speaker:
                f.write(f"Mono\t{start:.2f}\t{end:.2f}\n")
        for start, end, speakers in multi_segments:
            if target_speaker in speakers:
                f.write(f"Overlap\t{start:.2f}\t{end:.2f}\n")

def compute_similarity(reference_path, final_path):
    ref_embed = extract_embedding(reference_path, encoder)
    ext_embed = extract_embedding(final_path, encoder)
    similarity = 1 - cosine(ref_embed, ext_embed)
    return similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker extraction test script")
    parser.add_argument('--reference', type=str, required=True, help='Path to reference .wav file')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory with test .wav files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()

    REFERENCE_PATH = args.reference
    AUDIO_DIR = Path(args.audio_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for audio_file in AUDIO_DIR.glob("*.wav"):
        name = audio_file.stem
        current_output_dir = OUTPUT_DIR / name
        current_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔍 Обработка файла: {name}")

        # Диаризация
        mono_segments, multi_segments, diarization_result = run_diarization(str(audio_file))

        # Извлечение целевого спикера
        final_audio_path, target_speaker = extract_target_speaker(
            reference_path=REFERENCE_PATH,
            audio_path=str(audio_file),
            mono_segments=mono_segments,
            multi_segments=multi_segments,
            output_dir=current_output_dir,
            debug=True
        )

        # Метрики
        similarity = compute_similarity(REFERENCE_PATH, final_audio_path)
        print(f"🔗 Косинусное сходство (ref vs result): {similarity:.4f}")

        # Мелспектрограммы
        plot_melspectrogram(
            audio_path=str(audio_file),
            output_path=current_output_dir / f"{name}_mel_overlapped.png",
            title="Мелспектрограмма: многоспикерная запись"
        )

        plot_melspectrogram(
            audio_path=final_audio_path,
            output_path=current_output_dir / f"{name}_mel_extracted.png",
            title="Мелспектрограмма: извлечённый целевой спикер"
        )

        # Сохраняем результаты
        plot_waveform_comparison(
            original_path=str(audio_file),
            extracted_path=final_audio_path,
            output_path=current_output_dir / f"{name}_waveform_comparison.png"
        )

        plot_speaker_segments(
            mono_segments=mono_segments,
            multi_segments=multi_segments,
            target_speaker=target_speaker,
            output_path=current_output_dir / f"{name}_speaker_segments.png"
        )

        save_segment_list(
            mono_segments=mono_segments,
            multi_segments=multi_segments,
            target_speaker=target_speaker,
            output_path=current_output_dir / f"{name}_segments.txt"
        )

        with open(current_output_dir / f"{name}_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"Target speaker: {target_speaker}\n")
            f.write(f"Cosine similarity: {similarity:.4f}\n")

        print(f"✅ Готово: {current_output_dir}")
