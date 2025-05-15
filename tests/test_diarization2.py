# tests/test_diarization.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap

from models.diarization import run_diarization

# Добавим путь к корню проекта

OUTPUT_DIR = Path("tests/data/output/diarization_results")
AUDIO_DIR = Path("tests/data/audio/overlap_noise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def visualize_diarization(diarization_result, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.cm import get_cmap

    fig, ax = plt.subplots(figsize=(12, 2))

    # Определяем всех спикеров и их позиции по оси Y
    speakers = sorted(set(label for _, _, label in diarization_result.itertracks(yield_label=True)))
    speaker_y = {speaker: i for i, speaker in enumerate(speakers)}
    cmap = get_cmap("tab10")
    color_map = {speaker: cmap(i % 10) for i, speaker in enumerate(speakers)}  # один цвет на спикера

    # Рисуем сегменты
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        y = speaker_y[speaker]
        ax.add_patch(Rectangle(
            (turn.start, y - 0.4),
            turn.end - turn.start,
            0.8,
            facecolor=color_map[speaker],
            edgecolor='black'
        ))

    ax.set_yticks(list(speaker_y.values()))
    ax.set_yticklabels(speaker_y.keys())
    ax.set_xlabel("Время, сек")
    ax.set_title("Диаризация (все спикеры)")
    ax.set_xlim(0, max(turn.end for turn, _, _ in diarization_result.itertracks(yield_label=True)) + 1)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_diarization_to_text(diarization_result, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            f.write(f"{turn.start:.2f} - {turn.end:.2f}: {speaker}\n")

if __name__ == "__main__":
    for audio_file in AUDIO_DIR.glob("*.wav"):
        print(f"\n🎧 Обработка файла: {audio_file.name}")
        mono_segments, multi_segments, diarization_result = run_diarization(str(audio_file))

        print(f"💬 Сегментов одиночной речи: {len(mono_segments)}")
        print(f"💬 Сегментов с перекрытием: {len(multi_segments)}")

        base_name = audio_file.stem
        text_path = OUTPUT_DIR / f"{base_name}_diarization.txt"
        plot_path = OUTPUT_DIR / f"{base_name}_diarization.png"

        save_diarization_to_text(diarization_result, text_path)
        visualize_diarization(diarization_result, plot_path)

        print(f"✅ Результаты сохранены в {text_path} и {plot_path}")
