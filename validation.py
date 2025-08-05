import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        for idx, line in enumerate(lines):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"Ungültige Zeile {idx+1}: '{line.strip()}'")
                errors += 1
                continue
            input_text, expected_idx = parts
            if class_filter and expected_idx not in class_filter:
                continue
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
            if is_topn and threshold_ok:
                correct_topn += 1
            total += 1
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
            # Fortschritt mit Prozent und ETA
            if (idx+1) % 20 == 0 or (idx+1) == len(lines):
                percent = (idx+1)/len(lines)*100
                elapsed = time.time() - start_time
                eta = (elapsed/(idx+1))*(len(lines)-(idx+1)) if idx+1 < len(lines) else 0
                print(f"Fortschritt: {idx+1}/{len(lines)} ({percent:.1f}%) | ETA: {eta:.1f}s")
    if total > 0:
        print(f"Test-Genauigkeit (Top-1): {correct}/{total} = {correct/total*100:.1f}%")
        if top_n > 1:
            print(f"Test-Genauigkeit (Top-{top_n}): {correct_topn}/{total} = {correct_topn/total*100:.1f}%")
        if threshold is not None:
            print(f"Schwellenwert: {threshold}, gezählt nur wenn Wahrscheinlichkeit >= Schwelle")
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
    print("Konfusionsmatrix:")
    for true_cls in sorted(conf_matrix.keys()):
        print(f"{true_cls}: {conf_matrix[true_cls]}")
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
