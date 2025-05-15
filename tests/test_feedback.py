import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd

from pathlib import Path
from models.analysis import analyze_transcript
from models.feedback import generate_feedback

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY") # замени на свой ключ
INPUT_DIR = Path("tests/data/transcripts/")
OUTPUT_DIR = Path("tests/data/output/feedback/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_rows = []

for txt_file in INPUT_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")
    analysis = analyze_transcript(text)
    metrics = analysis["metrics"]

    print(f"📨 Отправляем текст: {txt_file.name}")
    try:
        feedback = generate_feedback(
            transcribed_text=text,
            total_words=metrics["Общее количество слов"],
            unique_words=metrics["Уникальных слов"],
            ttr=metrics["Type-Token Ratio (TTR)"],
            avg_sentence_length=metrics["Средняя длина предложения"],
            filler_counts=analysis["filler_counts"],
            api_key=API_KEY
        )

        feedback_path = OUTPUT_DIR / f"{txt_file.stem}_feedback.txt"
        feedback_path.write_text(feedback, encoding="utf-8")
        print(f"✅ Рекомендации сохранены: {feedback_path}")

        summary_rows.append({
            "Файл": txt_file.name,
            "Паразитов": metrics["Количество слов-паразитов"],
            "TTR": metrics["Type-Token Ratio (TTR)"],
            "Рекомендации": feedback[:120].replace("\n", " ") + "..."
        })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUTPUT_DIR / "feedback_summary.csv", index=False, encoding="utf-8")
        print("\n📊 Сводка рекомендаций сохранена.")


    except Exception as e:
        print(f"❌ Ошибка для {txt_file.name}: {e}")
