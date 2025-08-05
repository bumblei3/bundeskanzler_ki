import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def validate_model(tokenizer, model, maxlen, preprocess, detect_lang, test_file="test.txt"):
    if not os.path.exists(test_file):
        print(f"Keine Testdatei '{test_file}' gefunden. Validierung übersprungen.")
        return
    print(f"Validiere Modell mit Testdaten aus '{test_file}' ...")
    correct = 0
    total = 0
    with open(test_file, "r", encoding="utf-8") as ftest:
        for line in ftest:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            input_text, expected_idx = parts
            lang = detect_lang(input_text)
            input_pp = preprocess(input_text, lang=lang)
            input_seq = tokenizer.texts_to_sequences([input_pp])
            input_seq = pad_sequences(input_seq, maxlen=maxlen, padding='post')
            output = model.predict(input_seq)[0]
            pred_idx = int(np.argmax(output))
            if str(pred_idx) == expected_idx:
                correct += 1
            total += 1
    if total > 0:
        print(f"Test-Genauigkeit: {correct}/{total} = {correct/total*100:.1f}%")
    else:
        print("Keine gültigen Testdaten gefunden.")
