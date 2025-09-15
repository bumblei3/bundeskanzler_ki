import json
import os

import pytest
from corpus_manager import CorpusManager


def test_initialize_and_load_default(tmp_path):
    corpus_file = tmp_path / "corpus.json"
    # Datei existiert nicht, Standard-Korpus wird initialisiert
    manager = CorpusManager(str(corpus_file))
    # Korpus ist jetzt ein Dictionary mit Kategorie 'allgemein'
    assert isinstance(manager.corpus, dict)
    assert "allgemein" in manager.corpus
    assert len(manager.corpus["allgemein"]) == 3
    # Datei wurde angelegt
    assert corpus_file.exists()
    with open(corpus_file, encoding="utf-8") as f:
        data = json.load(f)
        assert "entries" in data
        assert len(data["entries"]) == 3


def test_add_sentence_and_get_all(tmp_path):
    corpus_file = tmp_path / "corpus.json"
    manager = CorpusManager(str(corpus_file))
    manager.add_sentence("Test-Satz", "Politik", "de")
    # get_all_sentences gibt in vereinfachter Version die Liste zurück
    all_sentences = manager.get_all_sentences()
    assert isinstance(all_sentences, list)
    assert any(
        "Test-Satz" in s
        for s in all_sentences
        if isinstance(s, str) or isinstance(s, dict)
    )


def test_save_and_reload(tmp_path):
    corpus_file = tmp_path / "corpus.json"
    manager = CorpusManager(str(corpus_file))
    manager.add_sentence("Noch ein Satz", "Wirtschaft", "de")
    manager.save_corpus()
    # Neu laden
    manager2 = CorpusManager(str(corpus_file))
    assert any(
        "Noch ein Satz" in s
        for s in manager2.get_all_sentences()
        if isinstance(s, str) or isinstance(s, dict)
    )
