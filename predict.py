import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Импортируем нашу функцию очистки из соседнего файла preprocess.py,
# чтобы обрабатывать новые твиты точно так же, как обучающие данные.
from preprocess import clean_tweet, setup_nltk

# Константы путей к файлам
MODEL_PATH = os.path.join('saved_models', 'best_model_glove.h5')
TOKENIZER_PATH = os.path.join('saved_models', 'tokenizer.pickle')
MAX_LEN = 50  # Должно совпадать с параметром при обучении!


def load_prediction_assets():
    """
    Загружает сохраненную модель (.h5) и токенизатор (.pickle).
    """
    print("Загрузка ресурсов для предсказания...")
    try:
        model = load_model(MODEL_PATH)

        # Токенизатор нужен, чтобы превратить слова пользователя в те же числа,
        # которые знает модель.
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)

        return model, tokenizer

    except FileNotFoundError:
        print("Ошибка: Файл модели или токенизатора не найден.")
        print(f"Убедитесь, что '{MODEL_PATH}' и '{TOKENIZER_PATH}' существуют.")
        print("Запустите main.py для их создания.")
        return None, None
    except Exception as e:
        print(f"Произошла ошибка при загрузке: {e}")
        return None, None


def predict_sentiment(text, model, tokenizer):
    """
    Основная логика инференса (предсказания):
    Raw Text -> Clean -> Tokenize -> Pad -> Predict -> Result
    """
    # 1. Очистка (убираем ссылки, стоп-слова, лемматизируем)
    cleaned_text = clean_tweet(text)

    # 2. Превращаем текст в последовательность чисел
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # 3. Дополняем нулями до длины 50 (padding)
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN,
                                    padding='post', truncating='post')

    # 4. Прогоняем через нейросеть
    prediction_prob = model.predict(padded_sequence)[0][0]

    # 5. Интерпретируем результат
    if prediction_prob > 0.5:
        return "POSITIVE", prediction_prob
    else:
        return "NEGATIVE", prediction_prob


def main():
    setup_nltk() # Убеждаемся, что базы NLTK скачаны

    model, tokenizer = load_prediction_assets()

    if model is None or tokenizer is None:
        print("Выход из программы.")
        return

    print("=" * 50)
    print("Модель для анализа тональности твитов готова.")
    print("Введи 'exit' или 'quit' для выхода.")
    print("=" * 50)

    # Бесконечный цикл для общения с пользователем
    while True:
        user_input = input("Введите твит для анализа: ")

        if user_input.lower() in ['exit', 'quit']:
            break

        if not user_input.strip():
            print("Пожалуйста, введите текст.")
            continue

        label, probability = predict_sentiment(user_input, model, tokenizer)

        if label == "POSITIVE":
            print(f"   -> РЕЗУЛЬТАТ: 🟢 ПОЗИТИВНЫЙ (Уверенность: {probability * 100:.1f}%)")
        else:
            # Для негативного уверенность = 1 - probability
            print(f"   -> РЕЗУЛЬТАТ: 🔴 НЕГАТИВНЫЙ (Уверенность: {(1 - probability) * 100:.1f}%)")
        print("-" * 30)


if __name__ == '__main__':
    main()