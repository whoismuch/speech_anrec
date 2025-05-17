import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import librosa

def identify_target_speaker(reference_path, audio_path, mono_segments, sample_rate=16000):
    print("🔜 Определение целевого спикера...")

    # Загружаем и кодируем эталонный голос
    encoder = VoiceEncoder()
    ref_wav = preprocess_wav(reference_path)
    ref_embed = encoder.embed_utterance(ref_wav)

    # Загружаем основное аудио
    y, sr = librosa.load(audio_path, sr=sample_rate)

    MIN_DURATION = 1.0  # в секундах

    speaker_embeddings = {}
    for start, end, speaker in mono_segments:
        if end - start < MIN_DURATION:
            continue

        segment_audio = y[int(start * sr):int(end * sr)]
        try:
            wav = preprocess_wav(segment_audio, source_sr=sr)
            embed = encoder.embed_utterance(wav)
            speaker_embeddings.setdefault(speaker, []).append(embed)
        except Exception as e:
            print(f"Ошибка сегмента {start:.2f}-{end:.2f}: {e}")

    # Усредняем эмбеддинги и сравниваем
    mean_embeddings = {s: np.mean(np.vstack(e), axis=0) for s, e in speaker_embeddings.items()}
    similarities = {s: cosine_similarity(ref_embed.reshape(1, -1), emb.reshape(1, -1))[0][0] for s, emb in mean_embeddings.items()}

    print("\n Сходство спикеров с эталоном:")
    for s, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {s}: {sim:.4f}")

    print("🔍 Similarities:", similarities)
    if not similarities or max(similarities.values()) < 0.75:
        print("⚠️ Нет подходящих сегментов для идентификации целевого спикера.")
        return (
            "NOT_FOUND",
            np.zeros_like(ref_embed),
            np.array([]),
            sr,
            encoder
        )

    target = max(similarities, key=similarities.get)
    print(f"\n ✅Целевой спикер: {target} (похожесть: {similarities[target]:.4f})")

    return target, ref_embed, y, sr, encoder

def get_encoder():
    return VoiceEncoder()

def extract_embedding(audio_path, encoder, sample_rate=16000):
    """
    Возвращает эмбеддинг аудиофайла с помощью переданного encoder.
    """
    wav = preprocess_wav(audio_path)
    return encoder.embed_utterance(wav)

