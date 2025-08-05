import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import argparse
import datetime
import csv
import yaml
import logging

def batch_inference(tokenizer: 'Tokenizer', model: 'tf.keras.Model', maxlen: int, corpus: list[str], corpus_original: list[str], args: argparse.Namespace) -> None:
    """
    Führt die Batch-Inferenz für eine Liste von Eingaben aus einer Datei durch und gibt die Ergebnisse aus.
    """
    logging.info(f"[Batch-Inferenz] Verarbeite Datei: {args.input}")
    batch_results = []
    total_lines = 0
    error_count = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    for idx_line, line in enumerate(lines, 1):
        seed_text = line.strip()
        total_lines += 1
        if not seed_text:
            logging.warning(f"[Batch-Inferenz] Leere Zeile {idx_line} übersprungen.")
            continue
        try:
            lang = detect_lang(seed_text)
            seed_text_pp = preprocess(seed_text, lang=lang)
            seed_sequence = tokenizer.texts_to_sequences([seed_text_pp])
            seed_sequence = pad_sequences(seed_sequence, maxlen=maxlen, padding='post')
            output = model.predict(seed_sequence)[0]
            top_indices = np.argsort(output)[::-1][:args.top_n]
            logging.info(f"[Batch-Inferenz] ({idx_line}/{len(lines)}) Eingabe: {seed_text}")
            logging.info(f"[Batch-Inferenz] Top-{args.top_n} Antworten für Eingabe: {seed_text}")
            antworten = []
            for i, idx in enumerate(top_indices):
                mark = "*" if i == 0 else " "
                logging.info(f"[Batch-Inferenz] {mark}{i+1}. {corpus[idx]} (Wahrscheinlichkeit: {output[idx]*100:.1f}%)")
                logging.info(f"[Batch-Inferenz]   Originalsatz: {corpus_original[idx]}")
                antworten.append((idx, output[idx]*100))
            log_interaction(seed_text, antworten, args.log, corpus, corpus_original)
            batch_results.append((seed_text, antworten))
        except Exception as e:
            error_count += 1
            print_error_hint(e)
    export_batch_results_csv(batch_results, corpus, corpus_original)
    logging.info(f"[Batch-Inferenz] Batch-Ergebnisse wurden als batch_results.csv exportiert.")
    logging.info(f"[Batch-Inferenz] Gesamt: {total_lines} Zeilen verarbeitet, Fehler: {error_count}")

def init_model(tokenizer: 'Tokenizer', maxlen: int, embedding_size: int = 256, lstm_output_size: int = 128) -> 'tf.keras.Model':
    """
    Initialisiert und gibt ein GRU-basiertes Keras-Modell zurück.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_size, input_length=maxlen),
        tf.keras.layers.GRU(lstm_output_size, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
    ])

def train_model(model: 'tf.keras.Model', X: np.ndarray, Y: np.ndarray, args: argparse.Namespace) -> 'tf.keras.Model':
    """
    Trainiert das Modell mit den gegebenen Daten und Parametern.
    """
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    model.fit(X, Y, batch_size=args.batch_size, epochs=args.epochs, callbacks=[early_stop])
    return model

def preprocess_corpus(corpus: list[str]) -> list[str]:
    """
    Führt Preprocessing und Spracherkennung für jeden Satz im Korpus durch.
    """
    corpus_pp = []
    for s in corpus:
        lang = detect_lang(s)
        corpus_pp.append(preprocess(s, lang=lang))
    return corpus_pp
# Globale Korpus-Variablen für Tests und andere Module
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
        logging.error("Datei nicht gefunden. Prüfe den Dateinamen und Pfad.")
    elif isinstance(e, ValueError):
        logging.error("Wertfehler: Prüfe die Eingabedaten und das Format.")
    elif isinstance(e, ImportError):
        logging.error("Importfehler: Prüfe, ob alle Pakete installiert sind (z.B. tensorflow, numpy, nltk, pandas, scikit-learn, streamlit, reportlab, pillow, matplotlib, langdetect).")
    else:
        logging.error(f"Unerwarteter Fehler: {e}")


def interactive_mode(tokenizer: 'Tokenizer', model: 'tf.keras.Model', maxlen: int, corpus: list[str], corpus_original: list[str], args: argparse.Namespace) -> None:
    """
    Startet den interaktiven Modus für Nutzereingaben und zeigt die Top-N Antworten an.
    """
    logging.info("[Interaktiv] Bundeskanzler-KI: Geben Sie eine Frage oder Aussage ein (Abbruch mit 'exit')")
    while True:
        seed_text = input("Ihre Eingabe: ")
        if seed_text.strip().lower() == "exit":
            logging.info("[Interaktiv] Sitzung beendet durch Nutzer.")
            break
        if not seed_text.strip():
            logging.warning("[Interaktiv] Leere Eingabe übersprungen.")
            continue
        # Validierung Modell und Tokenizer
        if model is None:
            print("Das Modell ist nicht geladen. Bitte prüfen Sie die Konfiguration und den Modellpfad.")
            logging.error("Interaktiv: Modell nicht geladen.")
            continue
        if tokenizer is None:
            print("Der Tokenizer ist nicht geladen. Bitte prüfen Sie die Konfiguration.")
            logging.error("Interaktiv: Tokenizer nicht geladen.")
            continue
        try:
            lang = detect_lang(seed_text)
            seed_text_pp = preprocess(seed_text, lang=lang)
            seed_sequence = tokenizer.texts_to_sequences([seed_text_pp])
            seed_sequence = pad_sequences(seed_sequence, maxlen=maxlen, padding='post')
            output = model.predict(seed_sequence)[0]
            top_indices = np.argsort(output)[::-1][:args.top_n]
            print(f"\nTop-{args.top_n} Antworten für Ihre Eingabe:")
            antworten = []
            for i, idx in enumerate(top_indices):
                mark = "*" if i == 0 else " "
                print(f"{mark}{i+1}. {corpus[idx]} (Wahrscheinlichkeit: {output[idx]*100:.1f}%)")
                print(f"   Originalsatz: {corpus_original[idx]}")
                antworten.append((idx, output[idx]*100))
            log_interaction(seed_text, antworten, args.log, corpus, corpus_original)
            feedback_interaction(seed_text, antworten, corpus)
        except Exception as e:
            print("Fehler bei der Verarbeitung Ihrer Eingabe. Details siehe Log.")
            print_error_hint(e)


def main():
    # Konfiguration laden
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Logging initialisieren
    log_file = config['data']['log_file']
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Kommandozeilenargumente für optionale Überschreibung
    parser = argparse.ArgumentParser(description="Bundeskanzler-KI")
    parser.add_argument('--top_n', type=int, help='Anzahl der Top-Antworten')
    parser.add_argument('--batch_size', type=int, help='Batchgröße für das Training')
    parser.add_argument('--epochs', type=int, help='Anzahl der Trainings-Epochen')
    parser.add_argument('--input', type=str, help='Datei für Batch-Inferenz')
    parser.add_argument('--corpus', type=str, help='Korpus-Datei')
    parser.add_argument('--log', type=str, help='Logdatei')
    args = parser.parse_args()

    # Parameter aus config.yaml, ggf. durch CLI überschrieben
    top_n = args.top_n if args.top_n is not None else config['general']['top_n']
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    input_file = args.input if args.input is not None else config['data']['input_file']
    corpus_file = args.corpus if args.corpus is not None else config['data']['corpus_file']
    log_file = args.log if args.log is not None else config['data']['log_file']
    embedding_size = config['model']['embedding_size']
    lstm_output_size = config['model']['lstm_output_size']
    maxlen = config['model']['maxlen']
    num_words = config['model']['num_words']
    model_path = config['model']['model_path']

    # Datensammlung
    corpus_original_local = []
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus_original_local.append(line)
        logging.info(f"Korpus aus Datei geladen: {len(corpus_original_local)} Sätze.")
    except FileNotFoundError:
        logging.warning(f"{corpus_file} nicht gefunden, Standardkorpus wird verwendet.")
        corpus_original_local = corpus_original.copy()
    corpus = corpus_original_local.copy()
    # Preprocess Korpus mit Spracherkennung
    corpus = preprocess_corpus(corpus)

    # Tokenisierung
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    X = pad_sequences(sequences, maxlen=maxlen, padding='post')

    # Definition des LSTM-Modells und One-hot-Encoding
    Y = np.eye(len(corpus))[np.arange(len(corpus))] # One-hot encoding der Targets

    if os.path.exists(model_path):
        logging.info("Lade vorhandenes Modell...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = init_model(tokenizer, maxlen, embedding_size, lstm_output_size)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # args für train_model zusammenbauen
        train_args = argparse.Namespace(batch_size=batch_size, epochs=epochs, top_n=top_n, input=input_file, corpus=corpus_file, log=log_file)
        model = train_model(model, X, Y, train_args)
        model.save(model_path)
        logging.info(f"Modell gespeichert unter {model_path}")

    # args für Inferenz/Interaktiv
    run_args = argparse.Namespace(batch_size=batch_size, epochs=epochs, top_n=top_n, input=input_file, corpus=corpus_file, log=log_file)

    # Batch-Inferenz: Eingaben aus Datei verarbeiten
    if os.path.exists(input_file):
        batch_inference(tokenizer, model, maxlen, corpus, corpus_original, run_args)
    else:
        interactive_mode(tokenizer, model, maxlen, corpus, corpus_original, run_args)

    # Validierungsfunktion für Testdaten
    validate_model(tokenizer, model, maxlen, preprocess, detect_lang)
    analyze_feedback()

if __name__ == "__main__":
    main()

