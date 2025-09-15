import csv
import datetime
import os


def log_interaction(eingabe, antworten, log_file, corpus, corpus_original):
    with open(log_file, "a", encoding="utf-8") as flog:
        flog.write(f"[{datetime.datetime.now().isoformat()}] Eingabe: {eingabe}\n")
        for i, (idx, wahrscheinlichkeit) in enumerate(antworten):
            flog.write(
                f"  {i+1}. {corpus[idx]} (Wahrscheinlichkeit: {wahrscheinlichkeit:.1f}%)\n"
            )
            flog.write(f"     Originalsatz: {corpus_original[idx]}\n")
        flog.write("\n")


def feedback_interaction(eingabe, antworten, corpus):
    feedback = input(
        "Feedback zu den Antworten (z.B. 1=korrekt, 2=falsch, 3=unpassend, Enter für Überspringen): "
    )
    with open("feedback.txt", "a", encoding="utf-8") as fback:
        fback.write(f"Eingabe: {eingabe}\n")
        fback.write(
            f"Antworten: {[(corpus[idx], wahrscheinlichkeit) for idx, wahrscheinlichkeit in antworten]}\n"
        )
        fback.write(f"Feedback: {feedback}\n\n")


def export_batch_results_csv(
    results, corpus, corpus_original, filename="batch_results.csv"
):
    with open(filename, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "Eingabe",
                "Antwort-Index",
                "Antwort",
                "Wahrscheinlichkeit",
                "Originalsatz",
            ]
        )
        for res in results:
            eingabe, antworten = res
            for i, (idx, wahrscheinlichkeit) in enumerate(antworten):
                writer.writerow(
                    [
                        eingabe,
                        idx,
                        corpus[idx],
                        wahrscheinlichkeit,
                        corpus_original[idx],
                    ]
                )


def analyze_feedback(feedback_file="feedback.txt"):
    if not os.path.exists(feedback_file):
        print("Keine Feedback-Datei gefunden.")
        return
    stats = {"korrekt": 0, "falsch": 0, "unpassend": 0, "leer": 0}
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Feedback:"):
                fb = line.strip().split(":", 1)[-1].strip()
                if fb == "":
                    stats["leer"] += 1
                elif "1" in fb:
                    stats["korrekt"] += 1
                elif "2" in fb:
                    stats["falsch"] += 1
                elif "3" in fb:
                    stats["unpassend"] += 1
    print("--- Feedback-Statistik ---")
    print(f"Korrekt:    {stats['korrekt']}")
    print(f"Falsch:     {stats['falsch']}")
    print(f"Unpassend:  {stats['unpassend']}")
    print(f"Übersprungen:{stats['leer']}")
    print("--------------------------")
