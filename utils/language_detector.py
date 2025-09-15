"""
Spracherkennung mit fastText für die Bundeskanzler-KI.
"""

import os
import tempfile
import urllib.request

import fasttext
import numpy as np


class LanguageDetector:
    """
    Spracherkennung mit fastText.
    Unterstützt die automatische Erkennung der Eingabesprache.
    """

    def __init__(self, model_path=None):
        """
        Initialisiert den Sprachdetektor.

        Args:
            model_path: Pfad zum vortrainierten Modell. Falls None, wird das Modell
                       automatisch heruntergeladen.
        """
        self.model = None
        self.model_path = model_path
        self.supported_languages = {"de", "en", "fr", "it", "es"}

    def download_model(self):
        """Lädt das vortrainierte fastText Spracherkennungsmodell herunter."""
        model_url = (
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        )

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            print("Lade Spracherkennungsmodell herunter...")
            urllib.request.urlretrieve(model_url, tmp_file.name)
            self.model_path = tmp_file.name
            print("Modell erfolgreich heruntergeladen.")

    def load_model(self):
        """Lädt das Spracherkennungsmodell."""
        if not self.model_path or not os.path.exists(self.model_path):
            self.download_model()

        print("Lade Spracherkennungsmodell...")
        self.model = fasttext.load_model(self.model_path)
        print("Modell erfolgreich geladen.")

    def detect_language(self, text, threshold=0.8):
        """
        Erkennt die Sprache eines Textes.

        Args:
            text: Der zu analysierende Text
            threshold: Minimaler Konfidenzwert für die Spracherkennung

        Returns:
            tuple: (Sprachcode, Konfidenz) oder ('unknown', 0.0) falls keine
                  unterstützte Sprache erkannt wurde
        """
        if not self.model:
            self.load_model()

        # Bereinige Text
        text = text.replace("\n", " ").strip()
        if not text:
            return "unknown", 0.0

        # Führe Spracherkennung durch
        predictions = self.model.predict(text, k=5)
        languages = [lang.replace("__label__", "") for lang in predictions[0]]
        confidences = np.asarray(predictions[1])

        # Finde die beste unterstützte Sprache
        for lang, conf in zip(languages, confidences):
            if lang in self.supported_languages and conf >= threshold:
                return lang, float(conf)

        return "unknown", 0.0

    def is_supported_language(self, lang_code):
        """Prüft, ob eine Sprache unterstützt wird."""
        return lang_code in self.supported_languages


# Beispielverwendung
if __name__ == "__main__":
    detector = LanguageDetector()

    test_texts = [
        "Das ist ein deutscher Text zur Überprüfung der Spracherkennung.",
        "This is an English text to test language detection.",
        "C'est un texte français pour tester la détection de la langue.",
        "Questo è un testo italiano per testare il rilevamento della lingua.",
        "Este es un texto en español para probar la detección de idioma.",
    ]

    for text in test_texts:
        lang, conf = detector.detect_language(text)
        print(f"\nText: {text[:50]}...")
        print(f"Erkannte Sprache: {lang}")
        print(f"Konfidenz: {conf:.2f}")
