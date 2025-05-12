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
from models.speaker_extraction import extract_target_speaker



from dotenv import load_dotenv
import os
import warnings

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
openrouter_key = os.getenv("OPENROUTER_API_KEY")


def main(audio_path, reference_path, output_dir, debug=False):
    warnings.filterwarnings("ignore")

    audio_path = Path(audio_path)
    basename = Path(audio_path).stem if debug else ""
    suffix = f"{basename}" if debug else ""

    reference_path = Path(reference_path)
    initial_output_dir = Path(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("🟢 Старт обработки...\n")

    # 1. Диаризация
    mono_segments, multi_segments, _ = run_diarization(str(audio_path))

    # 2-4. Извлечение целевого спикера (ID + Separation + Combine)
    combined_audio_path, target_speaker = extract_target_speaker(
        reference_path=str(reference_path),
        audio_path=str(audio_path),
        mono_segments=mono_segments,
        multi_segments=multi_segments,
        output_dir=output_dir,
        debug=debug
    )

    # 5. ASR
    transcript = transcribe_audio(str(combined_audio_path), model_size="base")
    os.makedirs(initial_output_dir/"transcript", exist_ok=True)
    output_dir = initial_output_dir / "transcript"
    transcript_path = output_dir / f"{suffix}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # 6. Анализ речи
    report = analyze_transcript(transcript)
    os.makedirs(initial_output_dir / "analysis_report", exist_ok=True)
    output_dir = initial_output_dir / "analysis_report"
    report_path = output_dir / f"{suffix}.md"
    save_report(report, path=str(report_path))

    # 7. AI-рекомендации
    ai_feedback = generate_feedback(
        transcribed_text=transcript,
        total_words=report["metrics"]["Общее количество слов"],
        unique_words=report["metrics"]["Уникальных слов"],
        ttr=report["metrics"]["Type-Token Ratio (TTR)"],
        avg_sentence_length=report["metrics"]["Средняя длина предложения"],
        filler_counts=report["filler_counts"],
        api_key=openrouter_key
    )
    os.makedirs(initial_output_dir/ "feedback", exist_ok=True)
    output_dir = initial_output_dir / "feedback"
    ai_feedback_path = output_dir / f"{suffix}.md"
    with open(ai_feedback_path, "w", encoding="utf-8") as f:
        f.write(ai_feedback)

    print("\n✅ Обработка завершена!")
    print(f"🎯 Target speaker: {target_speaker}")
    print(f"🎧 Cleaned audio: {combined_audio_path}")
    print(f"📝 Transcript: {transcript_path}")
    print(f"📊 Report: {report_path}")
    print(f"🤖 Feedback: {ai_feedback_path}")


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
