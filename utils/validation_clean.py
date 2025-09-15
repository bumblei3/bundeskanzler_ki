"""
Clean Validation Modul für die Bundeskanzler KI
Implementiert saubere Validierung ohne externe Abhängigkeiten
"""
import os
import numpy as np

# Sicherstellen, dass echte numpy verwendet wird
import sys
if 'numpy' in sys.modules and not hasattr(sys.modules['numpy'], 'linalg'):
    # Wahrscheinlich ein Stub, lade echtes numpy
    import numpy
    sys.modules['numpy'] = numpy
    np = numpy


def validate_model_clean(tokenizer, model, maxlen, preprocess, detect_lang, test_file):
    """
    Validiert das Modell mit sauberen Testdaten

    Args:
        tokenizer: Text tokenizer
        model: Das zu validierende Modell
        maxlen: Maximale Sequenzlänge
        preprocess: Preprocessing-Funktion
        detect_lang: Spracherkennungsfunktion
        test_file: Pfad zur Testdatei

    Returns:
        Validation results oder None bei Fehler
    """
    if not os.path.exists(test_file):
        return None

    try:
        # Testdaten laden
        texts = []
        labels = []

        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    text, label = line.strip().split('\t', 1)
                    texts.append(text)
                    labels.append(int(label))

        if not texts:
            return None

        # Preprocessing anwenden falls verfügbar
        if preprocess:
            texts = [preprocess(text) for text in texts]

        # Tokenisierung
        if tokenizer:
            sequences = tokenizer.texts_to_sequences(texts)
            # Padding auf maxlen
            padded_sequences = []
            for seq in sequences:
                if len(seq) > maxlen:
                    padded_sequences.append(seq[:maxlen])
                else:
                    padded_sequences.append(seq + [0] * (maxlen - len(seq)))
        else:
            # Fallback: einfache Sequenzen erstellen
            padded_sequences = [[i % 100 for i in range(min(len(text), maxlen))] +
                               [0] * (maxlen - min(len(text), maxlen)) for text in texts]

        # Vorhersagen treffen
        if model:
            predictions = model.predict(np.array(padded_sequences), verbose=0)

            # Accuracy berechnen
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == np.array(labels))

            # Zusätzliche Metriken
            results = {
                'accuracy': float(accuracy),
                'num_samples': len(texts),
                'predictions': pred_classes.tolist(),
                'true_labels': labels,
                'confidence_scores': np.max(predictions, axis=1).tolist()
            }

            # Spracherkennung falls verfügbar
            if detect_lang:
                languages = [detect_lang(text) for text in texts]
                results['languages'] = languages
                # Sprachbasierte Accuracy
                german_mask = np.array([lang == 'de' for lang in languages])
                if np.any(german_mask):
                    german_accuracy = np.mean(pred_classes[german_mask] == np.array(labels)[german_mask])
                    results['german_accuracy'] = float(german_accuracy)

            return results
        else:
            # Nur Datenstruktur zurückgeben ohne Vorhersagen
            return {
                'num_samples': len(texts),
                'texts': texts,
                'labels': labels,
                'sequences': padded_sequences
            }

    except Exception as e:
        print(f"Fehler bei der Validierung: {e}")
        return None


def validate_model_with_metrics(tokenizer, model, maxlen, test_file, metrics=None):
    """
    Erweiterte Validierung mit zusätzlichen Metriken

    Args:
        tokenizer: Text tokenizer
        model: Das zu validierende Modell
        maxlen: Maximale Sequenzlänge
        test_file: Pfad zur Testdatei
        metrics: Zusätzliche Metriken-Funktionen

    Returns:
        Validation results mit Metriken
    """
    results = validate_model_clean(tokenizer, model, maxlen, None, None, test_file)

    if results and metrics:
        # Zusätzliche Metriken berechnen
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])

        for metric_name, metric_func in metrics.items():
            try:
                results[metric_name] = float(metric_func(true_labels, predictions))
            except Exception as e:
                print(f"Fehler bei Metrik {metric_name}: {e}")
                results[metric_name] = None

    return results


def cross_validate_model(tokenizer, model, maxlen, test_files, folds=5):
    """
    Kreuzvalidierung über mehrere Testdateien

    Args:
        tokenizer: Text tokenizer
        model: Das zu validierende Modell
        maxlen: Maximale Sequenzlänge
        test_files: Liste von Testdateien
        folds: Anzahl der Folds (wird ignoriert, verwendet alle Dateien)

    Returns:
        Kreuzvalidierungsergebnisse
    """
    all_results = []

    for test_file in test_files:
        result = validate_model_clean(tokenizer, model, maxlen, None, None, test_file)
        if result:
            all_results.append(result)

    if not all_results:
        return None

    # Aggregierte Ergebnisse
    accuracies = [r['accuracy'] for r in all_results if 'accuracy' in r]
    num_samples = sum(r['num_samples'] for r in all_results)

    return {
        'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
        'std_accuracy': float(np.std(accuracies)) if accuracies else 0.0,
        'total_samples': num_samples,
        'num_folds': len(all_results),
        'fold_results': all_results
    }
