import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

def build_model(tokenizer, maxlen, embedding_size=256, lstm_output_size=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_size, input_length=maxlen),
        tf.keras.layers.GRU(lstm_output_size, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def load_or_train_model(tokenizer, X, Y, maxlen, args):
    model_path = "bundeskanzler_ki_model.keras"
    if os.path.exists(model_path):
        print("Lade vorhandenes Modell...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = build_model(tokenizer, maxlen)
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        model.fit(X, Y, batch_size=args.batch_size, epochs=args.epochs, callbacks=[early_stop])
        model.save(model_path)
        print(f"Modell gespeichert unter {model_path}")
    return model
