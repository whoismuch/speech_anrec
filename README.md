# Speech Feedback System

🧠 Интеллектуальная система для автоматического анализа устной речи и генерации персонализированных рекомендаций по её улучшению.

---

## ✨ Возможности

* Выделяет речь одного целевого говорящего из шумных и многоспикерных аудиозаписей
* Распознаёт речь с помощью Whisper
* Анализирует структуру речи: TTR, длина предложений, слова-паразиты и др.
* Генерирует рекомендации по улучшению речи с помощью LLM

---

## ⚡️ Быстрый старт

### 🔧 Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
pip install -r requirements.txt
```

### 🚀 Запуск пайплайна

```bash
python run_pipeline.py \
  --input data/input/main_audio.wav \
  --reference data/input/reference.wav \
  --output data/output \
  --hf_token YOUR_HF_TOKEN
```

### 🔑 Переменные окружения (опционально)

Создайте файл `.env` и добавьте:

```env
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 🔄 Архитектура

```
speech_feedback_system/
├── run_pipeline.py           # Главный скрипт запуска
├── requirements.txt
├── models/
│   ├── diarization.py        # Выделение спикеров (pyannote.audio)
│   ├── speaker_id.py         # Определение целевого говорящего
│   ├── separation.py         # SepFormer separation + speaker ID
│   ├── combine.py            # Объединение финального WAV
│   ├── asr.py                # Распознавание речи (Whisper)
│   ├── analysis.py           # Анализ текста речи
│   └── feedback.py           # AI-рекомендации через OpenRouter
├── data/
│   ├── input/                # main_audio.wav и reference.wav
│   └── output/               # итоговые WAV, текст и отчёты
```

---

## 💡 Пример результата

* `target_speaker_combined.wav` — речь целевого пользователя
* `transcript.txt` — текстовая расшифровка
* `feedback_report.md` — отчёт по метрикам речи
* `ai_feedback.md` — рекомендации от LLM

---

## 🚜 Дальнейшие шаги


---

## 📄 Лицензия

MIT License

---

Created by Khumai Bairamova, 2025
