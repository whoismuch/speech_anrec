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

    speaker_embeddings = {}
    for start, end, speaker in mono_segments:
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

    target = max(similarities, key=similarities.get)
    print(f"\n ✅Целевой спикер: {target} (похожесть: {similarities[target]:.4f})")

    return target, ref_embed, y, sr, encoder
