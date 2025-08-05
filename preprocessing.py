import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import GermanStemmer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set([
    "the", "is", "in", "and", "for", "will", "has", "that", "he", "she", "of", "an", "zu", "die", "der", "das", "ist", "und", "wird", "hat", "dass", "er", "sie", "in", "auf", "mit", "mehr", "als", "den", "dem", "des", "ein", "eine", "im", "am", "von", "auf", "zu", "für"
])

try:
    from langdetect import detect as langdetect_detect
except ImportError:
    langdetect_detect = None
    print("Warnung: langdetect nicht installiert. Erweiterte Spracherkennung deaktiviert.")

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
    # Nutze langdetect, falls verfügbar
    if langdetect_detect:
        try:
            lang_code = langdetect_detect(text)
            if lang_code.startswith('de'):
                return 'de'
            else:
                return 'en'
        except Exception:
            pass
    # Fallback: Sehr einfache Spracherkennung
    deutsch = ["kanzler", "regierung", "deutschland", "arbeitslosigkeit", "klimaschutz", "flüchtlinge", "wirtschaft", "bildung", "infrastruktur", "bundesregierung", "steuern"]
    if any(w in text.lower() for w in deutsch):
        return 'de'
    return 'en'
