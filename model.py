import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, SpatialDropout1D, Dense, Bidirectional
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

def build_model(vocab_size, embedding_dim, max_len, rnn_units, embedding_matrix):

    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=max_len,
                  weights=[embedding_matrix],
                  trainable=True),

        SpatialDropout1D(0.3),

        Bidirectional(GRU(units=rnn_units,
                          return_sequences=False,
                          dropout=0.2,
                          recurrent_dropout=0.2)),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )

    return model


class PredictionCallback(Callback):

    def __init__(self, X_test, y_test, tokenizer):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.tokenizer = tokenizer

        self.reverse_word_index = tokenizer.index_word

    def on_epoch_end(self, epoch, logs=None):
        indices = random.sample(range(len(self.X_test)), 3)

        print(f"\n--- Предсказания в конце эпохи {epoch + 1} ---")

        for i in indices:
            test_sample = self.X_test[i]
            true_label = self.y_test[i]

            words = [self.reverse_word_index.get(idx, '?') for idx in test_sample if idx != 0]
            original_text = " ".join(words)

            test_sample_batch = np.expand_dims(test_sample, axis=0)

            pred_prob = self.model.predict(test_sample_batch, verbose=0)[0][0]
            pred_label = 1 if pred_prob > 0.5 else 0

            print(f"  Текст: {original_text}")
            print(f"  Реальность: {'POSITIVE' if true_label == 1 else 'NEGATIVE'} | " \
                  f"Предсказание: {'POSITIVE' if pred_label == 1 else 'NEGATIVE'} (Prob: {pred_prob:.3f})")
        print("---------------------------------\n")