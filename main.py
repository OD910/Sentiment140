#main.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from preprocess import setup_nltk, clean_tweet
from model import build_model, PredictionCallback


DATA_FILE = os.path.join('data', 'training.1600000.processed.noemoticon.csv')
GLOVE_FILE = os.path.join('data', 'glove.twitter.27B.100d.txt')
LOG_DIR = 'logs'
MODEL_PATH = os.path.join('saved_models', 'best_model_glove.h5')
TOKENIZER_PATH = os.path.join('saved_models', 'tokenizer.pickle')

COLUMN_NAMES = ['target', 'id', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = "latin1"

MAX_VOCAB_SIZE = 20000
MAX_LEN = 50
EMBEDDING_DIM = 100
RNN_UNITS = 64

BATCH_SIZE = 1024
EPOCHS = 10
TEST_SPLIT_SIZE = 0.2

SAMPLE_SIZE = None


def load_glove_embeddings(glove_file_path):

    print(f"Загрузка векторов GloVe из {glove_file_path}...")
    embeddings_index = {}
    try:
        with open(glove_file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"ОШИБКА: Файл GloVe не найден по пути {glove_file_path}")
        print("Пожалуйста, скачайте 'glove.twitter.27B.100d.txt'")
        print("и поместите его в папку 'data'.")
        exit()
    print(f"Загружено {len(embeddings_index)} векторов слов.")
    return embeddings_index


def create_embedding_matrix(word_index, embeddings_index, embedding_dim, vocab_size):

    print("Создание матрицы весов Embedding...")
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    words_found = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            words_found += 1

    print(f"Матрица создана. Форма: {embedding_matrix.shape}")
    print(f"{words_found} из {vocab_size} слов найдено в GloVe.")
    return embedding_matrix


def main():
    setup_nltk()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    print(f"Загрузка данных из {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, encoding=DATASET_ENCODING, names=COLUMN_NAMES)

    df = df[['target', 'text']]
    df['target'] = df['target'].replace(4, 1)

    if SAMPLE_SIZE:
        print(f"Используем выборку из {SAMPLE_SIZE} записей для отладки.")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        print("Используем полный датасет (1.6 млн записей).")

    print("Распределение классов:")
    print(df['target'].value_counts())

    print("Начинаем очистку текста (это может занять время)...")
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    print("Очистка текста завершена.")

    print("Начинаем векторизацию текста...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_text'])

    print(f"Сохранение Tokenizer в {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer сохранен.")

    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN,
                                     padding='post', truncating='post')

    X = padded_sequences
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SPLIT_SIZE,
                                                        random_state=42,
                                                        stratify=y)

    print(f"Размер словаря (word_index): {len(tokenizer.word_index)}")
    print(f"Форма X_train: {X_train.shape}")
    print(f"Форма X_test: {X_test.shape}")

    embeddings_index = load_glove_embeddings(GLOVE_FILE)
    embedding_matrix = create_embedding_matrix(tokenizer.word_index,
                                               embeddings_index,
                                               EMBEDDING_DIM,
                                               MAX_VOCAB_SIZE)

    print("Собираем модель (с GloVe)...")
    model = build_model(vocab_size=MAX_VOCAB_SIZE,
                        embedding_dim=EMBEDDING_DIM,
                        max_len=MAX_LEN,
                        rnn_units=RNN_UNITS,
                        embedding_matrix=embedding_matrix)
    model.summary()

    print("Настраиваем Callback'и...")

    tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    early_stopping_callback = EarlyStopping(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            restore_best_weights=True)

    checkpoint_callback = ModelCheckpoint(filepath=MODEL_PATH,
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)

    prediction_cb = PredictionCallback(X_test, y_test, tokenizer)

    callbacks_list = [
        tensorboard_callback,
        early_stopping_callback,
        checkpoint_callback,
        prediction_cb
    ]

    print("=" * 50)
    print(f"Начинаем обучение на {SAMPLE_SIZE if SAMPLE_SIZE else 'ВСЕХ'} данных...")
    print("=" * 50)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )

    print("=" * 50)
    print("Обучение завершено.")
    print("=" * 50)

    print("Оценка лучшей модели (с GloVe) на тестовых данных:")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    print(f"\nМодель сохранена в: {MODEL_PATH}")
    print(f"Логи для TensorBoard находятся в: {LOG_DIR}")
    print(f"Чтобы запустить TensorBoard, введи в терминале: tensorboard --logdir {LOG_DIR}")


if __name__ == '__main__':
    main()