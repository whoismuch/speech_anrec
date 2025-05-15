import os

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
import pandas as pd
from pydub import AudioSegment
import whisper
from jiwer import wer
from tqdm import tqdm

# === Настройки ===
DATASET_PATH = Path("tests/data/cv-corpus-21.0-delta-2025-03-14/ru")  # ← замени на свой путь
N_SAMPLES = 10
MODELS = ["tiny", "base", "small", "medium", "large"]  # можно сократить
TEMP_DIR = Path("tests/data/temp/asr")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

WER_LOG_PATH = Path("tests/data/output/test_asr/asr_wer_results.txt")
WER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
WER_LOG_PATH.write_text("📊 WER по моделям:\n", encoding="utf-8")


# === Шаг 1: загружаем данные ===
df = pd.read_csv(DATASET_PATH / "validated.tsv", sep="\t")
subset = df[df["sentence"].notnull()].sample(n=N_SAMPLES, random_state=42)

# === Шаг 2: создаём пары .wav + референс ===
samples = []
for i, row in enumerate(subset.itertuples()):
    mp3_path = DATASET_PATH / "clips" / row.path
    wav_path = TEMP_DIR / f"sample{i}.wav"
    txt = row.sentence.strip().lower()

    # конвертация mp3 → wav
    audio = AudioSegment.from_mp3(mp3_path)
    audio.set_frame_rate(16000).export(wav_path, format="wav")

    samples.append((wav_path, txt))

# === Шаг 3: тестируем модели ===
results = []

for model_size in MODELS:
    print(f"\n🧪 Тестируем модель Whisper {model_size}")
    model = whisper.load_model(model_size)
    wers = []

    for wav_path, reference in tqdm(samples):
        result = model.transcribe(str(wav_path), language="ru")
        hypothesis = result["text"].strip().lower()
        sample_wer = wer(reference, hypothesis)
        wers.append(sample_wer)

    avg_wer = sum(wers) / len(wers)
    results.append((model_size, avg_wer))

    # Сохраняем результат в файл
    with open(WER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"- {model_size}: {avg_wer:.3f}\n")

# === Шаг 4: вывод результатов ===
print("\n📊 Сводка WER по моделям:")
print("| Model   | WER  |")
print("|---------|------|")
for model, score in results:
    print(f"| {model:<7} | {score:.3f} |")
