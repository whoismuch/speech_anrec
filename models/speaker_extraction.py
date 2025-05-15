# models/test_speaker_extraction.py

from pathlib import Path
from models.speaker_id import identify_target_speaker
from models.separation import run_separation
from models.combine import combine_segments
import os

def extract_target_speaker(reference_path, audio_path, mono_segments, multi_segments,
                           output_dir, debug=False):
    """
    Полный процесс извлечения целевого спикера:
    1. Speaker ID
    2. Separation overlapped сегментов
    3. Объединение всех фрагментов в WAV

    :param reference_path: путь к эталонному голосу
    :param audio_path: путь к многоспикерному аудио
    :param mono_segments: список (start, end, speaker) с одиночной речью
    :param multi_segments: список (start, end, [speakers]) с перекрытием
    :param output_dir: папка для сохранения результатов
    :param debug: добавлять постфикс имени входного файла
    :return: (путь к итоговому wav, имя target_speaker)
    """
    output_dir = Path(output_dir)
    basename = Path(audio_path).stem if debug else ""
    suffix = f"{basename}" if debug else ""

    # 1. Speaker ID
    target_speaker, ref_embed, y, sr, encoder = identify_target_speaker(
        reference_path=reference_path,
        audio_path=audio_path,
        mono_segments=mono_segments,
        sample_rate=16000
    )

    # 2. Separation
    target_segments = run_separation(
        y=y,
        sr=sr,
        multi_segments=multi_segments,
        target_speaker=target_speaker,
        ref_embed=ref_embed,
        encoder=encoder,
        output_dir=output_dir / "separated_segments"
    )

    # 3. Combine
    os.makedirs(output_dir/"target_speaker_combined", exist_ok=True)
    output_dir = output_dir / "target_speaker_combined"
    final_path = output_dir / f"{suffix}.wav"
    print("DEBUG FINAL PATH:", final_path)
    combine_segments(
        mono_segments=mono_segments,
        target_segments=target_segments,
        target_speaker=target_speaker,
        y=y,
        sr=sr,
        output_path=final_path
    )

    return final_path, target_speaker
