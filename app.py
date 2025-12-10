import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_tweet, setup_nltk

# --- КОНФИГУРАЦИЯ ---
app = Flask(__name__)

MODEL_PATH = os.path.join('saved_models', 'best_model_glove.h5')
TOKENIZER_PATH = os.path.join('saved_models', 'tokenizer.pickle')
MAX_LEN = 50

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None


def load_assets():
    """Загружаем модель и токенизатор один раз при старте сервера"""
    global model, tokenizer
    print("Загрузка NLTK...")
    setup_nltk()

    print("Загрузка модели и токенизатора...")
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Готово! Сервер ожидает запросы.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить файлы. {e}")


# Загружаем всё до первого запроса
load_assets()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    confidence = ""
    sentiment_color = ""
    input_text = ""

    if request.method == 'POST':
        # Получаем текст из формы
        input_text = request.form['text']

        if input_text.strip():
            # 1. Препроцессинг (как в твоем коде)
            cleaned_text = clean_tweet(input_text)

            # 2. Токенизация и паддинг
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

            # 3. Предсказание
            pred_prob = model.predict(padded_sequence)[0][0]

            # 4. Интерпретация
            if pred_prob > 0.5:
                prediction_text = "POSITIVE"
                confidence = f"{pred_prob * 100:.1f}%"
                # sentiment_color = "text-success"  # Зеленый класс Bootstrap
            else:
                prediction_text = "NEGATIVE"
                confidence = f"{(1 - pred_prob) * 100:.1f}%"
                # sentiment_color = "text-danger"  # Красный класс Bootstrap

    return render_template('index.html',
                           prediction=prediction_text,
                           confidence=confidence,
                           color=sentiment_color,
                           original_text=input_text)


if __name__ == '__main__':
    # debug=True позволяет видеть ошибки в браузере и перезагружает сервер при изменении кода
    app.run(debug=True)