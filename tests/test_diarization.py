# tests/test_diarization.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap

from models.diarization import run_diarization


def visualize_diarization(diarization_result):
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
    plt.show()


if __name__ == "__main__":
    audio_path = "tests/data/audio/overlapped_ali_db_practice_vasya_version_crutyak.wav"  # ← укажи нужный путь
    mono_segments, multi_segments, diarization_result = run_diarization(audio_path)

    print(f"\n💬 Сегментов одиночной речи: {len(mono_segments)}")
    print(f"💬 Сегментов с перекрытием: {len(multi_segments)}")

    visualize_diarization(diarization_result)
