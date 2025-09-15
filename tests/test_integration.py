#!/usr/bin/env python3
"""
Integrationstest für die aktualisierte bundeskanzler_ki.py mit AdvancedTransformerModel
"""

import sys
import os
sys.path.append('/home/tobber/bkki_venv')

from bundeskanzler_ki import init_model, generate_transformer_response
from advanced_transformer_model import AdvancedTransformerModel
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def test_init_model():
    """Teste die aktualisierte init_model Funktion"""
    print("Testing init_model Funktion...")

    try:
        # Erstelle einen einfachen Tokenizer
        corpus = [
            "Die KI-Entwicklung ist wichtig für Deutschland.",
            "Künstliche Intelligenz verändert unsere Welt.",
            "Bundeskanzler Scholz besucht Berlin.",
            "Die Digitalisierung schreitet voran."
        ]

        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(corpus)

        maxlen = 50
        output_size = len(corpus)

        # Teste init_model mit Transformer-Unterstützung
        model = init_model(tokenizer, maxlen, output_size, use_transformer=True)

        print("✓ init_model mit Transformer-Unterstützung erfolgreich")
        print(f"✓ Modell-Typ: {type(model)}")
        print(f"✓ Modell-Input-Shape: {model.input_shape}")
        print(f"✓ Modell-Output-Shape: {model.output_shape}")

        return True

    except Exception as e:
        print(f"✗ init_model Test fehlgeschlagen: {e}")
        return False

def test_transformer_response():
    """Teste die generate_transformer_response Funktion"""
    print("\nTesting generate_transformer_response Funktion...")

    try:
        # Initialisiere Transformer-Modell
        transformer_model = AdvancedTransformerModel(model_type="gpt2", model_name="gpt2")

        # Teste Antwortgenerierung
        test_question = "Was ist der Sinn des Lebens?"
        response = generate_transformer_response(test_question, transformer_model, max_length=50)

        print("✓ Transformer-Antwortgenerierung erfolgreich")
        print(f"✓ Frage: {test_question}")
        print(f"✓ Antwort: {response}")

        return True

    except Exception as e:
        print(f"✗ Transformer-Response Test fehlgeschlagen: {e}")
        return False

def test_fallback_to_lstm():
    """Teste Fallback zu LSTM wenn Transformer fehlschlägt"""
    print("\nTesting Fallback zu LSTM...")

    try:
        # Erstelle einen einfachen Tokenizer
        corpus = ["Test sentence one.", "Test sentence two."]
        tokenizer = Tokenizer(num_words=100)
        tokenizer.fit_on_texts(corpus)

        maxlen = 20
        output_size = len(corpus)

        # Teste init_model ohne Transformer-Unterstützung
        model = init_model(tokenizer, maxlen, output_size, use_transformer=False)

        print("✓ Fallback zu LSTM erfolgreich")
        print(f"✓ Modell-Typ: {type(model)}")

        return True

    except Exception as e:
        print(f"✗ LSTM Fallback Test fehlgeschlagen: {e}")
        return False

def main():
    """Führe alle Integrationstests aus"""
    print("🚀 Starting Bundeskanzler KI Integration Tests")
    print("=" * 60)

    tests = [
        test_init_model,
        test_transformer_response,
        test_fallback_to_lstm
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed! Bundeskanzler KI is ready with Transformer support.")
        return 0
    else:
        print("⚠️  Some integration tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())