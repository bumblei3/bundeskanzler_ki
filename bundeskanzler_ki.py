# Importiere TensorFlow Konfiguration
import tf_config

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

# Memory-Optimierung importieren
from memory_optimizer import (
    MemoryOptimizer,
    LazyFileReader,
    ChunkedProcessor,
    setup_memory_optimization,
    memory_optimizer
)

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from tqdm import tqdm

# Memory-Optimierung beim Start initialisieren
setup_memory_optimization()

def batch_inference(tokenizer: 'Tokenizer', model: 'tf.keras.Model', maxlen: int, corpus: list[str], corpus_original: list[str], args: argparse.Namespace) -> None:
    """
    Führt die Batch-Inferenz für eine Liste von Eingaben aus einer Datei durch.
    Optimiert: Batch-Prediction, optionales Ausgabeformat, Fehlerzusammenfassung.
    Memory-optimiert: Lazy Loading und Chunked Processing.
    """
    logging.info(f"[Batch-Inferenz] Verarbeite Datei: {args.input}")
    batch_results = []
    error_list = []
    total_lines = 0
    error_count = 0

    # Eingabedatei automatisch erstellen, falls sie fehlt
    if not os.path.exists(args.input):
        with open(args.input, "w", encoding="utf-8") as f:
            f.write("Testfrage\n")
        logging.warning(f"{args.input} nicht gefunden, Beispiel-Datei wurde erstellt.")
        return

    # Memory-optimierte Dateiverarbeitung
    file_reader = LazyFileReader(args.input)
    chunked_processor = ChunkedProcessor(chunk_size=500, memory_optimizer=memory_optimizer)

    # Sammle alle Zeilen für spätere Verarbeitung (aber nicht alles auf einmal laden)
    all_lines = list(file_reader.read_lines_lazy())
    total_lines = len(all_lines)

    logging.info(f"[Batch-Inferenz] {total_lines} Zeilen gefunden, starte Chunked-Verarbeitung")

    def process_chunk(chunk_lines):
        """Verarbeitet einen Chunk von Zeilen"""
        chunk_results = []
        chunk_errors = []

        for idx, seed_text in enumerate(chunk_lines):
            try:
                lang = detect_language(seed_text)
                seed_text_pp = preprocess(seed_text, lang=lang)

                if seed_text_pp:  # Nur nicht-leere verarbeiten
                    # Tokenisierung und Padding
                    sequence = tokenizer.texts_to_sequences([seed_text_pp])
                    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post')

                    # Prediction
                    output = model.predict(padded_sequence, verbose=0)[0]

                    # Top-N Antworten extrahieren
                    top_indices = np.argsort(output)[::-1][:args.top_n]
                    antworten = [(idx_ans, output[idx_ans]*100) for idx_ans in top_indices]

                    chunk_results.append((seed_text, antworten))

                    # Logging für diesen Eintrag
                    logging.info(f"[Batch-Inferenz] ({len(chunk_results)}/{len(chunk_lines)}) Eingabe: {seed_text}")
                    logging.info(f"[Batch-Inferenz] Top-{args.top_n} Antworten für Eingabe: {seed_text}")

                    for i, (idx_ans, score) in enumerate(antworten):
                        mark = "*" if i == 0 else " "
                        logging.info(f"[Batch-Inferenz] {mark}{i+1}. {corpus[idx_ans]} (Wahrscheinlichkeit: {score:.1f}%)")
                        logging.info(f"[Batch-Inferenz]   Originalsatz: {corpus_original[idx_ans]}")

                    if hasattr(args, 'print_answers') and args.print_answers:
                        print(f"\nTop-{args.top_n} Antworten für Eingabe: {seed_text}")
                        for i, (idx_ans, score) in enumerate(antworten):
                            mark = "*" if i == 0 else " "
                            print(f"{mark}{i+1}. {corpus[idx_ans]} (Wahrscheinlichkeit: {score:.1f}%)")
                            print(f"   Originalsatz: {corpus_original[idx_ans]}")

                    # Interaktion loggen
                    log_interaction(seed_text, antworten, args.log, corpus, corpus_original)

            except Exception as e:
                chunk_errors.append({'line': idx+1, 'input': seed_text, 'error': str(e)})
                print_error_hint(e)

        return chunk_results, chunk_errors

        # Chunked-Verarbeitung manuell implementieren für korrekte Fehlerbehandlung
        for i in range(0, len(all_lines), 500):  # Chunk size 500
            chunk_lines = all_lines[i:i+500]
            logging.debug(f"Verarbeite Chunk {i//500 + 1}/{(len(all_lines) + 499)//500}")

            # Verarbeite Chunk
            chunk_results, chunk_errors = process_chunk(chunk_lines)

            # Ergebnisse aggregieren
            batch_results.extend(chunk_results)
            error_list.extend(chunk_errors)
            error_count += len(chunk_errors)

            # Memory-Optimierung nach jedem Chunk
            memory_optimizer.force_garbage_collection()
            memory_optimizer.log_memory_usage(f"nach Chunk {i//500 + 1}")    # Finale Memory-Optimierung
    memory_optimizer.force_garbage_collection()
    memory_optimizer.log_memory_usage("nach Batch-Inferenz")

    # Zusammenfassung
    success_count = len(batch_results)
    logging.info(f"[Batch-Inferenz] Verarbeitung abgeschlossen: {success_count} erfolgreich, {error_count} Fehler")

    # Flexible Output-Dateinamen
    result_file = f"{getattr(args, 'output_path', '')}batch_results.{args.output_format}" if hasattr(args, 'output_format') else "batch_results.csv"
    error_file = f"{getattr(args, 'output_path', '')}batch_errors.{args.output_format if args.output_format == 'json' else 'csv'}"

    # Export
    if hasattr(args, 'output_format') and args.output_format == 'json':
        import json
        with open(result_file, 'w', encoding='utf-8') as fout:
            json.dump([
                {
                    'input': seed_text,
                    'antworten': [{'index': idx, 'score': score} for idx, score in antworten]
                } for seed_text, antworten in batch_results
            ], fout, ensure_ascii=False, indent=2)
        logging.info(f"[Batch-Inferenz] Ergebnisse als {result_file} exportiert.")

        # Fehlerzusammenfassung als JSON
        if error_list:
            with open(error_file, 'w', encoding='utf-8') as ferr:
                json.dump(error_list, ferr, ensure_ascii=False, indent=2)
            logging.info(f"[Batch-Inferenz] Fehler wurden als {error_file} exportiert.")
    else:
        # CSV Export
        export_batch_results_csv(batch_results, corpus, corpus_original)
        logging.info(f"[Batch-Inferenz] Batch-Ergebnisse wurden als {result_file} exportiert.")

        # Fehlerzusammenfassung als CSV
        if error_list:
            import csv
            with open(error_file, 'w', encoding='utf-8', newline='') as ferr:
                writer = csv.DictWriter(ferr, fieldnames=['line', 'input', 'error'])
                writer.writeheader()
                writer.writerows(error_list)
            logging.info(f"[Batch-Inferenz] Fehler wurden als {error_file} exportiert.")
        logging.info(f"[Batch-Inferenz] Batch-Ergebnisse wurden als {result_file} exportiert.")
    logging.info(f"[Batch-Inferenz] Gesamt: {total_lines} Zeilen verarbeitet, Fehler: {error_count}")

def init_model(tokenizer: 'Tokenizer', maxlen: int, output_size: int) -> 'tf.keras.Model':
    """
    Initialisiert und gibt ein Transformer-basiertes Modell zurück.
    
    Args:
        tokenizer: Der Tokenizer für die Texttransformation
        maxlen: Die maximale Länge der Eingabesequenzen
        output_size: Die Größe der Ausgabeschicht (Anzahl der Klassen)
    """
    from transformer_model import create_transformer_model
    
    vocab_size = len(tokenizer.word_index) + 1
    
    return create_transformer_model(
        maxlen=maxlen,
        vocab_size=vocab_size,
        output_size=output_size
    )

def train_model(model: 'tf.keras.Model', X: np.ndarray, Y: np.ndarray, args: argparse.Namespace) -> 'tf.keras.Model':
    """
    Trainiert das Modell mit den gegebenen Daten und Parametern.
    """
    from transformer_model import train_transformer
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            profile_batch='100,120'
        )
    ]
    
    history = train_transformer(
        model=model,
        X_train=X,
        y_train=Y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    return model

def preprocess_corpus(corpus: list[str]) -> list[str]:
    """
    Führt Preprocessing und Spracherkennung für jeden Satz im Korpus durch.
    """
    corpus_pp = []
    for s in corpus:
        lang = detect_language(s)
        corpus_pp.append(preprocess(s, lang=lang))
    return corpus_pp
# Korpus-Manager und Preprocessing importieren
from corpus_manager import CorpusManager

# Preprocessing auslagern
from preprocessing import preprocess
from language_detection import detect_language, get_supported_languages
# Feedback-Funktionen auslagern
from feedback import log_interaction, feedback_interaction, export_batch_results_csv, analyze_feedback

# Modell-Funktionen auslagern
from model import build_model, load_or_train_model
# Validierungsfunktion auslagern
from validation import validate_model


def print_error_hint(e):
    if isinstance(e, FileNotFoundError):
        logging.error("Datei nicht gefunden: Überprüfen Sie den Pfad und die Berechtigungen.")
    elif isinstance(e, ValueError):
        logging.error("Wertfehler: Überprüfen Sie die Eingabedaten und Parameter.")
    elif isinstance(e, ImportError):
        logging.error("Importfehler: Überprüfen Sie die Installation der erforderlichen Bibliotheken.")
    elif isinstance(e, RuntimeError):
        logging.error("Laufzeitfehler: Überprüfen Sie die Systemressourcen und Konfiguration.")
    elif isinstance(e, OSError):
        logging.error("Betriebssystemfehler: Überprüfen Sie Dateiberechtigungen und Systemressourcen.")
    else:
        logging.error(f"Unbekannter Fehler: {type(e).__name__}: {str(e)}")


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
            lang = detect_language(seed_text)
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
    # Konfiguration laden und validieren
    def validate_config(config):
        required = {
            'data': ['log_file', 'input_file', 'corpus_file'],
            'general': ['top_n'],
            'training': ['batch_size', 'epochs'],
            'model': ['embedding_size', 'lstm_output_size', 'maxlen', 'num_words', 'model_path']
        }
        missing = []
        for section, keys in required.items():
            if section not in config:
                missing.append(section)
                continue
            for key in keys:
                if key not in config[section]:
                    missing.append(f"{section}.{key}")
        if missing:
            print("Fehler: Fehlende Konfigurationswerte in config.yaml:")
            for m in missing:
                print(f"  - {m}")
            print("Bitte ergänze die fehlenden Werte und starte das Skript erneut.")
            exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)

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

    parser = argparse.ArgumentParser(description="Bundeskanzler-KI")
    parser.add_argument('--loglevel', type=str, default='INFO', help='Logging-Level: DEBUG, INFO, WARNING, ERROR')
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Trainiert das Modell")
    train_parser.add_argument('--epochs', type=int, help='Anzahl der Trainings-Epochen')
    train_parser.add_argument('--batch_size', type=int, help='Batchgröße für das Training')

    # Subcommand: infer
    infer_parser = subparsers.add_parser("infer", help="Batch-Inferenz aus Datei")
    infer_parser.add_argument('--input', type=str, help='Datei für Batch-Inferenz')
    infer_parser.add_argument('--top_n', type=int, help='Anzahl der Top-Antworten')
    infer_parser.add_argument('--output_format', type=str, choices=['csv', 'json'], default='csv', help='Exportformat: csv oder json')
    infer_parser.add_argument('--print_answers', action='store_true', help='Antworten direkt im Terminal ausgeben')
    infer_parser.add_argument('--output_path', type=str, default='', help='Pfad für Exportdateien (csv/json)')

    # Subcommand: interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interaktiver Modus")
    interactive_parser.add_argument('--top_n', type=int, help='Anzahl der Top-Antworten')

    # Subcommand: validate
    validate_parser = subparsers.add_parser("validate", help="Modellvalidierung")

    args = parser.parse_args()

    # Logging-Level setzen
    loglevel = getattr(args, 'loglevel', 'INFO').upper()
    loglevel_num = getattr(logging, loglevel, logging.INFO)
    logging.basicConfig(
        level=loglevel_num,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(config['data']['log_file'], encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Gemeinsame Parameter
    batch_size = getattr(args, 'batch_size', None) or config['training']['batch_size']
    epochs = getattr(args, 'epochs', None) or config['training']['epochs']
    top_n = getattr(args, 'top_n', None) or config['general']['top_n']
    input_file = getattr(args, 'input', None) or config['data']['input_file']
    corpus_file = config['data']['corpus_file']
    log_file = config['data']['log_file']
    embedding_size = config['model']['embedding_size']
    lstm_output_size = config['model']['lstm_output_size']
    maxlen = config['model']['maxlen']
    num_words = config['model']['num_words']
    model_path = config['model']['model_path']

    # Korpus laden
    corpus_manager = CorpusManager(corpus_file)
    corpus_original_local = corpus_manager.get_all_sentences()
    stats = corpus_manager.get_statistics()
    logging.info(f"Korpus geladen: {stats['total']} Sätze")
    for category, count in stats['by_category'].items():
        logging.info(f"  - {category}: {count} Sätze")
    for lang, count in stats['by_language'].items():
        logging.info(f"  - Sprache {lang}: {count} Sätze")
    corpus = corpus_original_local.copy()
    corpus_pp = preprocess_corpus(corpus)

    # Erstelle Training Daten
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(corpus_pp)
    sequences = tokenizer.texts_to_sequences(corpus_pp)
    X = pad_sequences(sequences, maxlen=maxlen, padding='post')
    Y = np.eye(len(corpus))[np.arange(len(corpus))]

    # Modell laden/erstellen
    output_size = len(corpus)  # Die Ausgabegröße sollte der Anzahl der Sätze im Korpus entsprechen
    vocab_size = len(tokenizer.word_index) + 1
    
    # Modell laden/erstellen
    if os.path.exists(model_path):
        logging.info("Lade vorhandenes Modell...")
        model = tf.keras.models.load_model(model_path)
        # Überprüfe, ob das geladene Modell die richtige Ausgabegröße hat
        if model.output_shape[-1] != output_size:
            logging.warning(f"Modell hat falsche Ausgabegröße {model.output_shape[-1]}, benötigt {output_size}. Erstelle neues Modell...")
            model = init_model(tokenizer, maxlen, output_size)
            model.compile(loss='categorical_crossentropy', optimizer='adam')
    else:
        model = init_model(tokenizer, maxlen, output_size)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Subcommand-Logik
    if args.command == "train":
        train_args = argparse.Namespace(batch_size=batch_size, epochs=epochs, top_n=top_n, input=input_file, corpus=corpus_file, log=log_file)
        model = train_model(model, X, Y, train_args)
        model.save(model_path)
        logging.info(f"Modell gespeichert unter {model_path}")
    elif args.command == "infer":
        run_args = argparse.Namespace(batch_size=batch_size, epochs=epochs, top_n=top_n, input=input_file, corpus=corpus_file, log=log_file, output_format=args.output_format, print_answers=args.print_answers, output_path=args.output_path)
        batch_inference(tokenizer, model, maxlen, corpus_pp, corpus, run_args)
    elif args.command == "interactive":
        run_args = argparse.Namespace(batch_size=batch_size, epochs=epochs, top_n=top_n, input=input_file, corpus=corpus_file, log=log_file)
        interactive_mode(tokenizer, model, maxlen, corpus_pp, corpus, run_args)
    elif args.command == "validate":
        validate_model(tokenizer, model, maxlen, preprocess, detect_language)
        analyze_feedback()
    

# Standard-Skriptstart
if __name__ == "__main__":
    main()

