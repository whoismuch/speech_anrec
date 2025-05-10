# models/feedback.py

import requests


def generate_feedback(transcribed_text: str, total_words: int, unique_words: int, ttr: float,
                      avg_sentence_length: float, filler_counts: dict, api_key: str) -> str:
    """
    Отправляет текст и метрики речи в OpenRouter API и получает рекомендации.
    """
    API_URL = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    prompt = f"""
        Ты — языковой ассистент, который помогает улучшить устную речь. Проанализируй следующие данные и выдай рекомендации на **русском языке**.
        
        Текст речи:
        \"\"\"\n{transcribed_text}\n\"\"\"
        
        Метрики речи:
        - Всего слов: {total_words}
        - Уникальных слов: {unique_words}
        - TTR: {ttr:.2f}
        - Средняя длина предложения: {avg_sentence_length:.2f}
        - Слов-паразитов: {sum(filler_counts.values())}
        - Найденные слова-паразиты: {', '.join(f'{k} ({v})' for k, v in filler_counts.items())}
        
        Сформулируй рекомендации по улучшению речи:
        1. Что стоит убрать или заменить?
        2. Как улучшить структуру и стиль?
        3. Как сделать речь более уверенной и выразительной?
        """

    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
