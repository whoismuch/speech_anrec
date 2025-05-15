import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pathlib import Path

from models.analysis import analyze_transcript, save_report
import pandas as pd

INPUT_DIR = Path("tests/data/transcripts/")
OUTPUT_DIR = Path("tests/data/output/text_analysis/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_rows = []

for txt_file in INPUT_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")
    report = analyze_transcript(text)

    # Сохраняем индивидуальный отчёт
    save_path = OUTPUT_DIR / f"{txt_file.stem}_report.txt"
    save_report(report, save_path)
    print(f"✅ Отчёт создан: {save_path}")

    # Сохраняем строку для таблицы
    metrics = report["metrics"]
    summary_rows.append({
        "Файл": txt_file.name,
        "Слов": metrics["Общее количество слов"],
        "Уникальных": metrics["Уникальных слов"],
        "TTR": metrics["Type-Token Ratio (TTR)"],
        "Длина предложения": metrics["Средняя длина предложения"],
        "Предложений": metrics["Количество предложений"],
        "Слов-паразитов": metrics["Количество слов-паразитов"],
        "Уникальных паразитов": metrics["Уникальных слов-паразитов"]
    })

# Создаём итоговую таблицу
summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "summary_table.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8")
print(f"\n📊 Таблица метрик сохранена: {summary_path}\n")

# Печать таблицы в консоль
print(summary_df.to_markdown(index=False))
