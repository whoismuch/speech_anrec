# run_pipeline.py

import argparse
from pathlib import Path

from models.diarization import run_diarization
from models.speaker_id import identify_target_speaker
from models.separation import run_separation
from models.combine import combine_segments
from models.asr import transcribe_audio
from models.analysis import analyze_transcript, save_report
from models.feedback import generate_feedback
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
openrouter_key = os.getenv("OPENROUTER_API_KEY")


def main(audio_path, reference_path, output_dir, debug=False):
    audio_path = Path(audio_path)
    basename = Path(audio_path).stem if debug else ""
    suffix = f"_{basename}" if debug else ""

    reference_path = Path(reference_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("🟢 Старт обработки...\n")

    # === 1. Диаризация
    mono_segments, multi_segments, diarization = run_diarization(str(audio_path), hf_token)

    # === 2. Определение целевого спикера
    target_speaker, ref_embed, y, sr, encoder = identify_target_speaker(
        reference_path=str(reference_path),
        audio_path=str(audio_path),
        mono_segments=mono_segments,
        sample_rate=16000
    )

    # === 3. Разделение перекрывающихся сегментов
    target_segments = run_separation(
        y=y,
        sr=sr,
        multi_segments=multi_segments,
        target_speaker=target_speaker,
        ref_embed=ref_embed,
        encoder=encoder,
        output_dir=output_dir / "separated_segments"
    )

    # === 4. Объединение всех сегментов
    final_path = output_dir / f"target_speaker_combined{suffix}.wav"
    combine_segments(
        mono_segments=mono_segments,
        target_segments=target_segments,
        target_speaker=target_speaker,
        y=y,
        sr=sr,
        output_path=final_path
    )

    print(f"\n✅ Готово! Аудио целевого спикера сохранено в: {final_path}")

    # === 5. Распознавание речи

    asr_path = output_dir / f"target_speaker_combined{suffix}.wav"
    transcript = transcribe_audio(str(asr_path), model_size="base")

    transcript_path = output_dir / f"transcript{suffix}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"\n📄 Текст сохранён в: {transcript_path}")

    # === 6. Анализ текста

    report = analyze_transcript(transcript)

    report_path = output_dir / f"analysis_report{suffix}.md"
    save_report(report, path=str(report_path))

    print(f"\n📊 Речевой анализ сохранён: {report_path}")

    # === 6. Генерация рекомендаций
    ai_feedback = generate_feedback(
        transcribed_text=transcript,
        total_words=report["metrics"]["Общее количество слов"],
        unique_words=report["metrics"]["Уникальных слов"],
        ttr=report["metrics"]["Type-Token Ratio (TTR)"],
        avg_sentence_length=report["metrics"]["Средняя длина предложения"],
        filler_counts=report["filler_counts"],
        api_key=openrouter_key
    )

    # Сохраняем в файл
    with open(output_dir / f'feedback{suffix}.md', "w", encoding="utf-8") as f:
        f.write(ai_feedback)

    print(f"\n🧠 Рекомендации сохранены в: {output_dir /  f'feedback{suffix}.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Feedback Pipeline")
    parser.add_argument("--input", required=True, help="Path to main audio file (wav)")
    parser.add_argument("--reference", required=True, help="Path to reference speaker audio (wav)")
    parser.add_argument("--output", default="data/output", help="Output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Включить режим отладки (добавляет постфиксы к результатам)")

    args = parser.parse_args()

    main(
        audio_path=args.input,
        reference_path=args.reference,
        output_dir=args.output,
        debug=args.debug
    )
