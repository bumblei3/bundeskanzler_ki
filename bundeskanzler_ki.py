import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import argparse

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


import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import GermanStemmer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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


# Stoppwort-Liste (deutsch/englisch, minimal)
STOPWORDS = set([
    "the", "is", "in", "and", "for", "will", "has", "that", "he", "she", "of", "an", "zu", "die", "der", "das", "ist", "und", "wird", "hat", "dass", "er", "sie", "in", "auf", "mit", "mehr", "als", "den", "dem", "des", "ein", "eine", "im", "am", "von", "auf", "zu", "für"
])


def preprocess(text, lang='de'):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    # Lemmatization/Stemming
    if lang == 'de':
        stemmer = GermanStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def detect_lang(text):
    # Sehr einfache Spracherkennung: Wenn viele deutsche Wörter, dann 'de', sonst 'en'
    deutsch = ["kanzler", "regierung", "deutschland", "arbeitslosigkeit", "klimaschutz", "flüchtlinge", "wirtschaft", "bildung", "infrastruktur", "bundesregierung", "steuern"]
    if any(w in text.lower() for w in deutsch):
        return 'de'
    return 'en'

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




# Logging-Funktion
import datetime

def log_interaction(eingabe, antworten):
    with open(args.log, "a", encoding="utf-8") as flog:
        flog.write(f"[{datetime.datetime.now().isoformat()}] Eingabe: {eingabe}\n")
        for i, (idx, wahrscheinlichkeit) in enumerate(antworten):
            flog.write(f"  {i+1}. {corpus[idx]} (Wahrscheinlichkeit: {wahrscheinlichkeit:.1f}%)\n")
            flog.write(f"     Originalsatz: {corpus_original[idx]}\n")
        flog.write("\n")

# Batch-Inferenz: Eingaben aus Datei verarbeiten
if os.path.exists(args.input):
    print(f"Batch-Inferenz: Verarbeite {args.input} ...")
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
                log_interaction(seed_text, antworten)
                print()
            except Exception as e:
                print(f"Fehler bei Eingabe '{seed_text}': {e}")
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
            log_interaction(seed_text, antworten)
        except Exception as e:
            print(f"Fehler: {e}")
