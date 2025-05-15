import re
from collections import Counter
import pandas as pd

def analyze_transcript(text: str):
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    total_words = len(words)
    unique_words = len(set(words))
    ttr = unique_words / total_words if total_words else 0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = sum(len(re.findall(r"\b\w+\b", s)) for s in sentences) / len(sentences) if sentences else 0

    word_counts = Counter(words)
    most_common = word_counts.most_common(10)

    filler_words = {
        "ну", "как бы", "короче", "в общем", "по сути", "на самом деле",
        "вроде бы", "это самое", "типа", "значит", "получается",
        "вот", "эээ", "ммм", "как это", "то есть", "скажем", "вот это",
        "так сказать", "в общем-то", "вообще", "получается так",
        "ещё бы", "такой", "так", "просто", "реально", "типа того",
        "всё такое", "ну да", "ну вот", "ну типа", "и всё такое"
    }

    filler_counts = {word: count for word, count in word_counts.items() if word in filler_words}

    metrics = {
        "Общее количество слов": total_words,
        "Уникальных слов": unique_words,
        "Type-Token Ratio (TTR)": round(ttr, 2),
        "Средняя длина предложения": round(avg_sentence_length, 2),
        "Количество предложений": len(sentences),
        "Количество слов-паразитов": sum(filler_counts.values()),
        "Уникальных слов-паразитов": len(filler_counts)
    }

    return {
        "metrics": metrics,
        "filler_counts": filler_counts,
        "most_common": most_common
    }

def save_report(report_data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("## Базовые метрики\n")
        for k, v in report_data["metrics"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Слова-паразиты\n")
        for w, c in report_data["filler_counts"].items():
            f.write(f"- {w}: {c} раз\n")
        f.write("\n## Топ-10 частотных слов\n")
        for w, c in report_data["most_common"]:
            f.write(f"- {w}: {c} раз\n")
