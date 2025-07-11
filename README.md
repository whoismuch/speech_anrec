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

Добавьте флаг --debug, чтобы сохранять результаты с постфиксом имени входного файла:

python run_pipeline.py \
  --input data/input/meeting1.wav \
  --reference data/input/ali.wav \
  --output data/output \
  --debug

Файлы будут называться, например: transcript_meeting1.txt, feedback_meeting1.md и т.п.



### 🔑 Переменные окружения (опционально)

Создайте файл `.env` и добавьте:

```env
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 🚀 Быстрый старт с conda-окружением

1. Склонируйте репозиторий и перейдите в папку проекта.
2. Создайте окружение из файла environment.yml:

```bash
conda env create -f environment.yml
```

3. Активируйте окружение:

```bash
conda activate speech_anrec
```

Теперь все зависимости будут установлены автоматически, и вы сможете запускать пайплайн и тесты без дополнительных настроек.

---

## 🧪 Тестирование модуля выделения целевого говорящего

Для запуска теста используйте:

```bash
python tests/test_speaker_extraction.py \
  --reference tests/data/audio/reference/ali_imba_drink.wav \
  --audio_dir tests/data/audio/ali/test3 \
  --output_dir tests/data/output/speaker_extraction_results_FINAL_FINAL_THESIS_THESIS/ali
```

- `--reference` — путь к эталонному .wav файлу целевого спикера
- `--audio_dir` — папка с тестовыми .wav файлами для обработки
- `--output_dir` — папка для сохранения результатов

Результаты (метрики, графики, сегменты) будут сохранены в указанной папке для каждого тестового файла.

---

## 🔄 Архитектура

```
speech_anrec/
├── run_pipeline.py           # Главный скрипт запуска
├── requirements.txt
├── environment.yml           # Окружение conda
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
├── tests/                    # Тесты и тестовые данные
```

---

## 💡 Пример результата

- `target_speaker_combined.wav` — речь целевого пользователя
- `transcript.txt` — текстовая расшифровка
- `analysis_report.md` — отчёт по метрикам речи
- `ai_feedback.md` — рекомендации от LLM

---

## 📄 Лицензия

MIT License

---

Created by Khumai Bairamova, 2025
