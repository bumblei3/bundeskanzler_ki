import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sicherstellen, dass echte numpy verwendet wird
import sys
if hasattr(np, 'exp') and not hasattr(np, 'linalg'):
    # Wahrscheinlich ein Stub, lade echtes numpy
    import numpy
    sys.modules['numpy'] = numpy
    np = numpy

def validate_model(tokenizer, model, maxlen, preprocess, detect_lang, test_file="test.txt", csv_out=None, top_n=1, log_file=None, threshold=None, class_filter=None, return_df=False):
    """
    Validiert das Modell mit Testdaten. Optional: CSV-Ausgabe, Top-N-Genauigkeit, Logging.
    Gibt Statistiken als dict zurück.
    """
    import csv
    import time
    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    import json
    if not os.path.exists(test_file):
        print(f"Keine Testdatei '{test_file}' gefunden. Validierung übersprungen.")
        return None
    print(f"Validiere Modell mit Testdaten aus '{test_file}' ...")
    correct = 0
    correct_topn = 0
    total = 0
    errors = 0
    results = []
    with open(test_file, "r", encoding="utf-8") as ftest:
        lines = ftest.readlines()
        start_time = time.time()
        conf_matrix = {}
        bar_len = 40
        y_true = []
        y_pred = []
        probs_correct = []
        probs_incorrect = []
        class_counts = {}
        for idx, line in enumerate(lines):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"Ungültige Zeile {idx+1}: '{line.strip()}'")
                errors += 1
                continue
            input_text, expected_idx = parts
            if class_filter and expected_idx not in class_filter:
                continue
            class_counts[expected_idx] = class_counts.get(expected_idx, 0) + 1
            lang = detect_lang(input_text)
            input_pp = preprocess(input_text, lang=lang)
            input_seq = tokenizer.texts_to_sequences([input_pp])
            input_seq = pad_sequences(input_seq, maxlen=maxlen, padding='post')
            try:
                output = model.predict(input_seq, verbose=0)[0]
            except Exception as e:
                print(f"Fehler bei Modellvorhersage in Zeile {idx+1}: {e}")
                errors += 1
                continue
            pred_idx = int(np.argmax(output))
            topn_indices = np.argsort(output)[-top_n:][::-1]
            prob_pred = float(output[pred_idx])
            is_correct = str(pred_idx) == expected_idx
            is_topn = str(expected_idx) in [str(i) for i in topn_indices]
            threshold_ok = True if threshold is None else prob_pred >= threshold
            if is_correct and threshold_ok:
                correct += 1
                probs_correct.append(prob_pred)
            else:
                probs_incorrect.append(prob_pred)
            if is_topn and threshold_ok:
                correct_topn += 1
            total += 1
            y_true.append(expected_idx)
            y_pred.append(str(pred_idx))
            results.append({
                "input": input_text,
                "expected": expected_idx,
                "predicted": str(pred_idx),
                "topn": [str(i) for i in topn_indices],
                "probs": [float(output[i]) for i in topn_indices],
                "is_correct": is_correct,
                "is_topn": is_topn,
                "prob_pred": prob_pred,
                "threshold_ok": threshold_ok
            })
            # Konfusionsmatrix
            conf_matrix.setdefault(expected_idx, {})
            conf_matrix[expected_idx].setdefault(str(pred_idx), 0)
            conf_matrix[expected_idx][str(pred_idx)] += 1
            # ASCII-Fortschrittsbalken
            if (idx+1) % 10 == 0 or (idx+1) == len(lines):
                percent = (idx+1)/len(lines)
                filled_len = int(bar_len * percent)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)
                elapsed = time.time() - start_time
                eta = (elapsed/(idx+1))*(len(lines)-(idx+1)) if idx+1 < len(lines) else 0
                print(f"[{bar}] {percent*100:.1f}% | ETA: {eta:.1f}s", end='\r' if (idx+1)!=len(lines) else '\n')
    if total > 0:
        print(f"Test-Genauigkeit (Top-1): {correct}/{total} = {correct/total*100:.1f}%")
        if top_n > 1:
            print(f"Test-Genauigkeit (Top-{top_n}): {correct_topn}/{total} = {correct_topn/total*100:.1f}%")
        if threshold is not None:
            print(f"Schwellenwert: {threshold}, gezählt nur wenn Wahrscheinlichkeit >= Schwelle")
        # Durchschnittliche Wahrscheinlichkeiten
        if probs_correct:
            print(f"Durchschnittliche Wahrscheinlichkeit (korrekt): {np.mean(probs_correct):.4f}")
        if probs_incorrect:
            print(f"Durchschnittliche Wahrscheinlichkeit (inkorrekt): {np.mean(probs_incorrect):.4f}")
        # Warnung bei seltenen Klassen
        few_samples = [cls for cls, cnt in class_counts.items() if cnt < 3]
        if few_samples:
            print(f"Warnung: Folgende Klassen haben weniger als 3 Testdaten: {few_samples}")
    else:
        print("Keine gültigen Testdaten gefunden.")
    if csv_out:
        with open(csv_out, "w", encoding="utf-8", newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=["input", "expected", "predicted", "topn", "probs", "is_correct", "is_topn", "prob_pred", "threshold_ok"])
            writer.writeheader()
            for r in results:
                r2 = r.copy()
                r2["topn"] = ",".join(r2["topn"])
                r2["probs"] = ",".join([f"{p:.4f}" for p in r2["probs"]])
                writer.writerow(r2)
        print(f"Ergebnisse als CSV gespeichert: {csv_out}")
    if log_file:
        with open(log_file, "a", encoding="utf-8") as flog:
            flog.write(f"Validierung {test_file}: Top-1={correct}, Top-{top_n}={correct_topn}, Total={total}, Fehler={errors}, Schwelle={threshold}\n")
    # Konfusionsmatrix ausgeben
    print("\nKonfusionsmatrix:")
    for true_cls in sorted(conf_matrix.keys()):
        print(f"{true_cls}: {conf_matrix[true_cls]}")
    # Visualisierung der Konfusionsmatrix (optional)
    if has_matplotlib:
        try:
            all_classes = sorted(conf_matrix.keys())
            pred_classes = sorted({p for d in conf_matrix.values() for p in d.keys()})
            matrix = np.zeros((len(all_classes), len(pred_classes)), dtype=int)
            for i, true_cls in enumerate(all_classes):
                for j, pred_cls in enumerate(pred_classes):
                    matrix[i, j] = conf_matrix[true_cls].get(pred_cls, 0)
            plt.figure(figsize=(8,6))
            plt.imshow(matrix, cmap='Blues')
            plt.xticks(range(len(pred_classes)), pred_classes, rotation=90)
            plt.yticks(range(len(all_classes)), all_classes)
            plt.xlabel('Vorhergesagt')
            plt.ylabel('Wahr')
            plt.title('Konfusionsmatrix')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            print("Konfusionsmatrix als Bild gespeichert: confusion_matrix.png")
            plt.show()
        except Exception as e:
            print(f"Fehler bei Konfusionsmatrix-Plot: {e}")
    # Fehleranalyse: häufigste Verwechslungen
    print("\nHäufigste Verwechslungen:")
    confusion_pairs = {}
    for true_cls, preds in conf_matrix.items():
        for pred_cls, count in preds.items():
            if true_cls != pred_cls:
                confusion_pairs[(true_cls, pred_cls)] = count
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    for (true_cls, pred_cls), count in top_confusions:
        print(f"{true_cls} -> {pred_cls}: {count}x")
    # Top-N Fehlerbeispiele (niedrigste Wahrscheinlichkeit für richtige Klasse)
    print("\nTop-Fehlerbeispiele (niedrigste Wahrscheinlichkeit für richtige Klasse):")
    error_examples = [r for r in results if not r["is_correct"]]
    error_examples = sorted(error_examples, key=lambda r: r["prob_pred"])[:5]
    for r in error_examples:
        print(f"Input: {r['input']} | Erwartet: {r['expected']} | Vorhergesagt: {r['predicted']} | Prob_richtig: {r['prob_pred']:.4f}")
    # Zusätzliche Metriken (F1, Precision, Recall)
    if has_sklearn and y_true and y_pred:
        try:
            f1 = f1_score(y_true, y_pred, average='weighted')
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            print(f"F1-Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        except Exception as e:
            print(f"Fehler bei Metrik-Berechnung: {e}")
    # Optional: JSON-Export
    try:
        with open('validation_results.json', 'w', encoding='utf-8') as jf:
            json.dump(results, jf, ensure_ascii=False, indent=2)
        print("Ergebnisse als JSON gespeichert: validation_results.json")
    except Exception as e:
        print(f"Fehler beim JSON-Export: {e}")
    result_dict = {
        "total": total,
        "correct": correct,
        "correct_topn": correct_topn,
        "errors": errors,
        "accuracy": correct/total if total else 0.0,
        "accuracy_topn": correct_topn/total if total else 0.0,
        "results": results,
        "conf_matrix": conf_matrix
    }
    if return_df and has_pandas:
        return pd.DataFrame(results)
    return result_dict
