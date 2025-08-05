import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import argparse
import datetime
import csv

# Preprocessing auslagern
from preprocessing import preprocess, detect_lang
# Feedback-Funktionen auslagern
from feedback import log_interaction, feedback_interaction, export_batch_results_csv, analyze_feedback
# Modell-Funktionen auslagern
from model import build_model, load_or_train_model
# Validierungsfunktion auslagern
from validation import validate_model

def print_error_hint(e):
    if isinstance(e, FileNotFoundError):
        print("Datei nicht gefunden. Prüfe den Dateinamen und Pfad.")
    elif isinstance(e, ValueError):
        print("Wertfehler: Prüfe die Eingabedaten und das Format.")
    elif isinstance(e, ImportError):
        print("Importfehler: Prüfe, ob alle Pakete installiert sind (z.B. tensorflow, numpy, nltk).")
    else:
        print(f"Unerwarteter Fehler: {e}")

def main():
    # Kommandozeilenargumente für konfigurierbare Parameter
    parser = argparse.ArgumentParser(description="Bundeskanzler-KI")
    parser.add_argument('--top_n', type=int, default=3, help='Anzahl der Top-Antworten')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchgröße für das Training')
    parser.add_argument('--epochs', type=int, default=50, help='Anzahl der Trainings-Epochen')
    parser.add_argument('--input', type=str, default='input.txt', help='Datei für Batch-Inferenz')
    parser.add_argument('--corpus', type=str, default='corpus.txt', help='Korpus-Datei')
    parser.add_argument('--log', type=str, default='log.txt', help='Logdatei')
    args = parser.parse_args()

    # Datensammlung
    # Erweiterter Korpus mit deutschen und englischen Sätzen
    corpus_original = []
    try:
        with open(args.corpus, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus_original.append(line)
        print(f"Korpus aus Datei geladen: {len(corpus_original)} Sätze.")
    except FileNotFoundError:
        print(f"{args.corpus} nicht gefunden, Standardkorpus wird verwendet.")
        corpus_original = [
            "The chancellor said he will increase taxes for the rich.",
            "The German chancellor believes that the economy is growing.",
            "Angela Merkel has announced that Germany will take in more refugees.",
            "Der Kanzler sagt, er wird die Steuern für Reiche erhöhen.",
            "Die deutsche Kanzlerin glaubt, dass die Wirtschaft wächst.",
            "Angela Merkel hat angekündigt, dass Deutschland mehr Flüchtlinge aufnehmen wird.",
            "Die Regierung plant Investitionen in Bildung und Infrastruktur.",
            "Die Arbeitslosigkeit in Deutschland ist gesunken.",
            "Die Inflation bleibt stabil.",
            "Die Bundesregierung diskutiert über neue Klimaschutzmaßnahmen."
        ]
    corpus = corpus_original.copy()

    # Preprocess Korpus mit Spracherkennung
    corpus_pp = []
    for s in corpus:
        lang = detect_lang(s)
        corpus_pp.append(preprocess(s, lang=lang))
    corpus = corpus_pp

    # Tokenisierung
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    maxlen = 120
    X = pad_sequences(sequences, maxlen=maxlen, padding='post')

    # Definition des LSTM-Modells
    embedding_size = 256
    lstm_output_size = 128
    Y = np.eye(len(corpus))[np.arange(len(corpus))] # One-hot encoding der Targets

    model_path = "bundeskanzler_ki_model.keras"
    if os.path.exists(model_path):
        print("Lade vorhandenes Modell...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_size, input_length=maxlen),
            tf.keras.layers.GRU(lstm_output_size, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(len(corpus), activation='softmax')
        ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # EarlyStopping Callback
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

    # Nur trainieren, wenn Modell nicht geladen wurde
    if not os.path.exists(model_path):
        model.fit(X, Y, batch_size=args.batch_size, epochs=args.epochs, callbacks=[early_stop])
        model.save(model_path)
        print(f"Modell gespeichert unter {model_path}")

    # Batch-Inferenz: Eingaben aus Datei verarbeiten
    if os.path.exists(args.input):
        print(f"Batch-Inferenz: Verarbeite {args.input} ...")
        batch_results = []
        with open(args.input, "r", encoding="utf-8") as fin:
            for line in fin:
                seed_text = line.strip()
                if not seed_text:
                    continue
                try:
                    lang = detect_lang(seed_text)
                    seed_text_pp = preprocess(seed_text, lang=lang)
                    seed_sequence = tokenizer.texts_to_sequences([seed_text_pp])
                    seed_sequence = pad_sequences(seed_sequence, maxlen=maxlen, padding='post')
                    output = model.predict(seed_sequence)[0]
                    top_indices = np.argsort(output)[::-1][:args.top_n]
                    print(f"Eingabe: {seed_text}")
                    print(f"Top-{args.top_n} Antworten:")
                    antworten = []
                    for i, idx in enumerate(top_indices):
                        mark = "*" if i == 0 else " "
                        print(f"{mark}{i+1}. {corpus[idx]} (Wahrscheinlichkeit: {output[idx]*100:.1f}%)")
                        print(f"   Originalsatz: {corpus_original[idx]}")
                        antworten.append((idx, output[idx]*100))
                    log_interaction(seed_text, antworten, args.log, corpus, corpus_original)
                    batch_results.append((seed_text, antworten))
                    print()
                except Exception as e:
                    print_error_hint(e)
        export_batch_results_csv(batch_results, corpus, corpus_original)
        print("Batch-Ergebnisse wurden als batch_results.csv exportiert.")
    else:
        # Verbesserte interaktive Eingabe: Preprocessing, Prozentwerte, Endlosschleife
        print("Bundeskanzler-KI: Geben Sie eine Frage oder Aussage ein (Abbruch mit 'exit')")
        while True:
            seed_text = input("Ihre Eingabe: ")
            if seed_text.strip().lower() == "exit":
                print("Beendet.")
                break
            try:
                lang = detect_lang(seed_text)
                seed_text_pp = preprocess(seed_text, lang=lang)
                seed_sequence = tokenizer.texts_to_sequences([seed_text_pp])
                seed_sequence = pad_sequences(seed_sequence, maxlen=maxlen, padding='post')
                output = model.predict(seed_sequence)[0]
                top_indices = np.argsort(output)[::-1][:args.top_n]
                print(f"Top-{args.top_n} Antworten:")
                antworten = []
                for i, idx in enumerate(top_indices):
                    mark = "*" if i == 0 else " "
                    print(f"{mark}{i+1}. {corpus[idx]} (Wahrscheinlichkeit: {output[idx]*100:.1f}%)")
                    print(f"   Originalsatz: {corpus_original[idx]}")
                    antworten.append((idx, output[idx]*100))
                log_interaction(seed_text, antworten, args.log, corpus, corpus_original)
                feedback_interaction(seed_text, antworten, corpus)
            except Exception as e:
                print_error_hint(e)
    # Validierungsfunktion für Testdaten
    validate_model(tokenizer, model, maxlen, preprocess, detect_lang)
    analyze_feedback()

if __name__ == "__main__":
    main()

