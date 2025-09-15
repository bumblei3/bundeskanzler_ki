import os
import tempfile
from feedback import log_interaction, export_batch_results_csv, analyze_feedback

def test_log_interaction(tmp_path):
    log_file = tmp_path / "log.txt"
    eingabe = "Testfrage"
    antworten = [(0, 99.9), (1, 50.0)]
    corpus = ["Antwort1", "Antwort2"]
    corpus_original = ["Antwort1", "Antwort2"]
    log_interaction(eingabe, antworten, str(log_file), corpus, corpus_original)
    with open(log_file, encoding='utf-8') as f:
        content = f.read()
        assert "Eingabe: Testfrage" in content
        assert "Antwort1" in content
        assert "Wahrscheinlichkeit: 99.9%" in content

def test_export_batch_results_csv(tmp_path):
    filename = tmp_path / "batch_results.csv"
    results = [("Frage1", [(0, 80.0), (1, 20.0)])]
    corpus = ["Antwort1", "Antwort2"]
    corpus_original = ["Antwort1", "Antwort2"]
    export_batch_results_csv(results, corpus, corpus_original, filename=str(filename))
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
        assert "Eingabe" in lines[0]
        assert "Frage1" in lines[1]
        assert "Antwort1" in lines[1]

def test_analyze_feedback(tmp_path, capsys):
    feedback_file = tmp_path / "feedback.txt"
    with open(feedback_file, "w", encoding="utf-8") as f:
        f.write("Feedback: 1\nFeedback: 2\nFeedback: 3\nFeedback: \n")
    analyze_feedback(str(feedback_file))
    captured = capsys.readouterr()
    assert "Korrekt:    1" in captured.out
    assert "Falsch:     1" in captured.out
    assert "Unpassend:  1" in captured.out
    assert "Übersprungen:1" in captured.out
