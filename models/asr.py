# models/asr.py

import whisper

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Распознаёт речь в аудиофайле с помощью Whisper.

    :param audio_path: путь к WAV-файлу
    :param model_size: "tiny", "base", "small", "medium", "large"
    :return: расшифрованный текст
    """
    print(f"📝 Распознаём речь (модель: {model_size})...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    print("✅ Распознавание завершено.")
    return result["text"]
