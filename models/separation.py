from speechbrain.pretrained import SepformerSeparation as separator
from resemblyzer import preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import soundfile as sf
import numpy as np
import torchaudio

def run_separation(y, sr, multi_segments, target_speaker, ref_embed, encoder, output_dir="separated_segments"):
    print("🔀 Запуск separation по перекрывающимся сегментам...")

    os.makedirs(output_dir, exist_ok=True)
    separation_model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir="pretrained_models/sepformer-whamr")

    target_segments = []

    for idx, (start, end, speakers_in_turn) in enumerate(multi_segments):
        if target_speaker not in speakers_in_turn:
            continue

        print(f"\n🎧 Сегмент {idx}: {start:.2f}-{end:.2f} (целевой участвует)")
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_tensor = torch.tensor(y[start_sample:end_sample]).unsqueeze(0)
        temp_input = f"{output_dir}/segment_{idx}_input.wav"
        torchaudio.save(temp_input, segment_tensor, sr)

        est_sources = separation_model.separate_file(path=temp_input)  # (1, samples, 2)
        source1 = est_sources[0, :, 0].detach().cpu().numpy()
        source2 = est_sources[0, :, 1].detach().cpu().numpy()

        path1 = f"{output_dir}/segment_{idx}_stream_0.wav"
        path2 = f"{output_dir}/segment_{idx}_stream_1.wav"
        sf.write(path1, source1, 8000, subtype="PCM_16")
        sf.write(path2, source2, 8000, subtype="PCM_16")

        best_sim, best_path = -1, None
        for i, path in enumerate([path1, path2]):
            try:
                wav = preprocess_wav(path, source_sr=8000)
                embed = encoder.embed_utterance(wav)
                sim = cosine_similarity(ref_embed.reshape(1, -1), embed.reshape(1, -1))[0][0]
                print(f"     Поток {i}: similarity = {sim:.4f}")
                if sim > best_sim:
                    best_sim = sim
                    best_path = path
            except Exception as e:
                print(f"     ⚠️ Ошибка ID потока {i}: {e}")

        if best_sim > 0.5:
            print(f"     ✅ Выбран поток: {best_path} (sim={best_sim:.4f})")
            target_segments.append((best_path, start, end))
        else:
            print("     🚫 Ни один поток не прошёл порог.")

    return target_segments
