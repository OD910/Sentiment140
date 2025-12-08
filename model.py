import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, SpatialDropout1D, Dense, Bidirectional
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

def build_model(vocab_size, embedding_dim, max_len, rnn_units, embedding_matrix):
    """
    Создает и компилирует модель нейронной сети.
    Аргументы:
      vocab_size: размер словаря (сколько всего слов знает модель).
      embedding_dim: размер вектора слова (например, 100 для GloVe-100d).
      max_len: длина последовательности (обрезаем или дополняем твиты до 50 слов).
      rnn_units: количество нейронов в GRU слое.
      embedding_matrix: готовая матрица весов из GloVe.
    """
    model = Sequential([
        # 1. Слой Embeddings.
        # Превращает индексы слов в плотные векторы.
        # weights=[embedding_matrix] загружает предобученные знания (GloVe).
        # trainable=True означает, что мы разрешаем модели "дообучать" эти векторы под нашу задачу.
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=max_len,
                  weights=[embedding_matrix],
                  trainable=True),

        # 2. SpatialDropout1D.
        # В отличие от обычного Dropout, этот слой "выключает" целые карты признаков (каналы),
        # а не отдельные элементы. Это эффективнее для обработки текста (снижает переобучение).
        SpatialDropout1D(0.3),

        # 3. Bidirectional GRU.
        # GRU (Gated Recurrent Unit) - упрощенная версия LSTM.
        # Bidirectional означает, что сеть читает твит и слева направо, и справа налево,
        # чтобы лучше понять контекст.
        Bidirectional(GRU(units=rnn_units,
                          return_sequences=False, # Возвращаем только последний вектор состояния, а не всю последовательность
                          dropout=0.2,            # Дропаут на входе линейных трансформаций
                          recurrent_dropout=0.2)), # Дропаут на рекуррентных связях

        # 4. Выходной слой.
        # Один нейрон с активацией sigmoid выдает число от 0 до 1.
        # 0 -> Негатив, 1 -> Позитив.
        Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    # Используем Adam с небольшим learning_rate для более плавной сходимости.
    model.compile(
        loss='binary_crossentropy', # Стандартная функция потерь для бинарной классификации
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )

    return model


class PredictionCallback(Callback):
    """
    Кастомный коллбэк для Keras.
    В конце каждой эпохи берет 3 случайных твита из тестовой выборки
    и показывает, как модель их классифицирует прямо сейчас.
    Это полезно, чтобы глазами видеть прогресс, а не только сухие цифры loss/accuracy.
    """
    def __init__(self, X_test, y_test, tokenizer):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.tokenizer = tokenizer
        # Создаем обратный словарь {индекс: слово} для расшифровки твитов обратно в текст
        self.reverse_word_index = tokenizer.index_word

    def on_epoch_end(self, epoch, logs=None):
        # Выбираем 3 случайных индекса
        indices = random.sample(range(len(self.X_test)), 3)

        print(f"\n--- Предсказания в конце эпохи {epoch + 1} ---")

        for i in indices:
            test_sample = self.X_test[i]
            true_label = self.y_test[i]

            # Декодируем последовательность чисел обратно в слова (игнорируем 0 - это padding)
            words = [self.reverse_word_index.get(idx, '?') for idx in test_sample if idx != 0]
            original_text = " ".join(words)

            # Добавляем размерность batch (модель ожидает [batch_size, sequence_length])
            test_sample_batch = np.expand_dims(test_sample, axis=0)

            # Делаем предсказание
            pred_prob = self.model.predict(test_sample_batch, verbose=0)[0][0]
            pred_label = 1 if pred_prob > 0.5 else 0

            print(f"  Текст: {original_text}")
            print(f"  Реальность: {'POSITIVE' if true_label == 1 else 'NEGATIVE'} | " \
                  f"Предсказание: {'POSITIVE' if pred_label == 1 else 'NEGATIVE'} (Prob: {pred_prob:.3f})")
        print("---------------------------------\n")