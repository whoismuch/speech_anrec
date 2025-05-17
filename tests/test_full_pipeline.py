
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from models.diarization import run_diarization
from models.speaker_id import identify_target_speaker
from models.speaker_extraction import extract_target_speaker
from models.separation import run_separation
from models.combine import combine_segments
from models.asr import transcribe_audio
from models.analysis import analyze_transcript, save_report
from models.feedback import generate_feedback
from test_speaker_extraction import (
    plot_waveform_comparison,
    plot_speaker_segments,
    plot_melspectrogram,
    save_segment_list
)
import os
from dotenv import load_dotenv

load_dotenv()
openrouter_key = os.getenv("OPENROUTER_API_KEY")

INPUT_AUDIO = "tests/data/audio/overlapped_bad_russian_ali_db_practice_progress.wav"
REFERENCE = "tests/data/audio/reference/bad_russian_reference.wav"
OUTDIR = Path("tests/data/output/full_test/")
OUTDIR.mkdir(exist_ok=True, parents=True)

print("🔍 Диаризация...")
mono_segments, multi_segments, diarization_result = run_diarization(INPUT_AUDIO)
save_segment_list(mono_segments, multi_segments, "auto", OUTDIR / "segments.txt")

print("🎯 Определение целевого...")
combined_audio_path, target_speaker = extract_target_speaker(
    reference_path=REFERENCE,
    audio_path=INPUT_AUDIO,
    mono_segments=mono_segments,
    multi_segments=multi_segments,
    output_dir=OUTDIR,
    debug=True
)

print("🎼 Построение спектрограмм...")
plot_melspectrogram(INPUT_AUDIO, OUTDIR / "mel_orig.png", "Мелспектрограмма: оригинал")
plot_melspectrogram(combined_audio_path, OUTDIR / "mel_cleaned.png", "Мелспектрограмма: целевой спикер")
plot_waveform_comparison(INPUT_AUDIO, combined_audio_path, OUTDIR / "waveform_comparison.png")
plot_speaker_segments(mono_segments, multi_segments, target_speaker, OUTDIR / "segments_vis.png")

print("🔠 Распознавание речи...")
transcript = transcribe_audio(str(combined_audio_path), model_size="small")
transcript_path = OUTDIR / "transcript.txt"
transcript_path.write_text(transcript, encoding="utf-8")

print("📊 Анализ речи...")
analysis = analyze_transcript(transcript)
save_report(analysis, OUTDIR / "analysis_report.txt")

print("🤖 AI-рекомендации...")
feedback = generate_feedback(
    transcribed_text=transcript,
    total_words=analysis["metrics"]["Общее количество слов"],
    unique_words=analysis["metrics"]["Уникальных слов"],
    ttr=analysis["metrics"]["Type-Token Ratio (TTR)"],
    avg_sentence_length=analysis["metrics"]["Средняя длина предложения"],
    filler_counts=analysis["filler_counts"],
    api_key=openrouter_key
)
(OUTDIR / "feedback.txt").write_text(feedback, encoding="utf-8")

print("\n✅ Тест завершён. Все артефакты сохранены в:", OUTDIR)
