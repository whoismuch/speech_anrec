# models/diarization.py

from pyannote.audio import Pipeline
import os

hf_token = os.getenv("HF_TOKEN")

def run_diarization(audio_path: str):
    """
    Выполняет диаризацию аудиофайла.

    :param audio_path: путь к аудиофайлу (wav)
    :return: mono_segments, multi_segments, full diarization object
    """
    print("Запуск диаризации...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    diarization = pipeline(audio_path)

    mono_segments = []
    multi_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
        overlapping = diarization.crop(turn, mode="loose").labels()
        if len(overlapping) == 1:
            mono_segments.append((turn.start, turn.end, speaker))
        else:
            multi_segments.append((turn.start, turn.end, overlapping))  # сохраним список спикеров

    print(f"Диаризация завершена. mono: {len(mono_segments)}, multi: {len(multi_segments)}")
    return mono_segments, multi_segments, diarization
