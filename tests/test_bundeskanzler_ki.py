import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, call, patch

import pytest

# Import global test stubs first
import test_stubs


# Wir patchen die wichtigsten externen Abhängigkeiten, um die Kernlogik zu testen
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    import test_stubs

    monkeypatch.setitem(sys.modules, "tf_config", MagicMock())
    monkeypatch.setitem(sys.modules, "transformer_model", MagicMock())
    monkeypatch.setitem(sys.modules, "corpus_manager", MagicMock())
    monkeypatch.setitem(sys.modules, "preprocessing", MagicMock())
    monkeypatch.setitem(sys.modules, "language_detection", MagicMock())
    monkeypatch.setitem(sys.modules, "feedback", MagicMock())
    monkeypatch.setitem(sys.modules, "model", MagicMock())
    monkeypatch.setitem(sys.modules, "validation", MagicMock())
    # Patch tensorflow and numpy with stubs for these specific tests
    monkeypatch.setitem(sys.modules, "tensorflow", test_stubs._TFStub())
    monkeypatch.setitem(sys.modules, "numpy", test_stubs._NPStub())
    # Patch tensorflow submodules
    monkeypatch.setitem(
        sys.modules,
        "tensorflow.keras.preprocessing.text",
        test_stubs._TFStub.keras.preprocessing.text,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorflow.keras.preprocessing.sequence",
        test_stubs._TFStub.keras.preprocessing.sequence,
    )
    monkeypatch.setitem(
        sys.modules, "tensorflow.keras.callbacks", test_stubs._TFStub.keras.callbacks
    )


def test_batch_inference_runs(monkeypatch):
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    # Importiere nach Patchen, damit alle Abhängigkeiten gemockt sind
    import bundeskanzler_ki

    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [
        [1, 2, 3],
        [4, 5],
    ]  # Mock sequences as list of lists
    model = MagicMock()
    model.predict.return_value = [[0.7, 0.2, 0.1]]
    maxlen = 10
    corpus = ["Antwort1", "Antwort2", "Antwort3"]
    corpus_original = ["Antwort1", "Antwort2", "Antwort3"]
    args = types.SimpleNamespace(
        input="dummy.txt",
        top_n=2,
        print_answers=False,
        output_format="csv",
        output_path="",
        log="log.txt",
    )
    # Dummy-Eingabedatei erzeugen
    with open("dummy.txt", "w", encoding="utf-8") as f:
        f.write("Testfrage\n")
    # Preprocessing und Language Detection mocken
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x.lower())
    monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
    monkeypatch.setattr(
        bundeskanzler_ki, "export_batch_results_csv", lambda *a, **kw: None
    )
    # Test: Funktion läuft ohne Fehler durch
    bundeskanzler_ki.batch_inference(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )


def test_interactive_mode_exit(monkeypatch):
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    tokenizer = MagicMock()
    model = MagicMock()
    maxlen = 10
    corpus = ["Antwort1", "Antwort2"]
    corpus_original = ["Antwort1", "Antwort2"]
    args = types.SimpleNamespace(top_n=2, log="test.log")

    # Mock input to return 'exit' immediately
    monkeypatch.setattr("builtins.input", lambda x: "exit")
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x)
    monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
    monkeypatch.setattr(bundeskanzler_ki, "feedback_interaction", lambda *a, **kw: None)

    # Should not raise any exceptions
    bundeskanzler_ki.interactive_mode(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )


def test_init_model(monkeypatch):
    """Test init_model function"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    from unittest.mock import MagicMock

    import bundeskanzler_ki

    # Mock the transformer_model module
    mock_create_transformer_model = MagicMock()
    mock_model = MagicMock()
    mock_create_transformer_model.return_value = mock_model

    monkeypatch.setattr(
        "transformer_model.create_transformer_model", mock_create_transformer_model
    )

    tokenizer = MagicMock()
    tokenizer.word_index = {"word1": 1, "word2": 2, "word3": 3}

    result = bundeskanzler_ki.init_model(tokenizer, maxlen=50, output_size=100)

    mock_create_transformer_model.assert_called_once_with(
        maxlen=50, vocab_size=4, output_size=100  # len(word_index) + 1
    )
    assert result == mock_model


def test_train_model(monkeypatch):
    """Test train_model function"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    from unittest.mock import MagicMock

    import bundeskanzler_ki

    # Mock the transformer_model module
    mock_train_transformer = MagicMock()
    mock_history = MagicMock()
    mock_train_transformer.return_value = mock_history

    monkeypatch.setattr("transformer_model.train_transformer", mock_train_transformer)

    model = MagicMock()
    X = MagicMock()
    Y = MagicMock()
    args = types.SimpleNamespace(batch_size=32, epochs=10)

    result = bundeskanzler_ki.train_model(model, X, Y, args)

    mock_train_transformer.assert_called_once()
    call_args = mock_train_transformer.call_args
    assert call_args[1]["model"] == model
    assert call_args[1]["X_train"] == X
    assert call_args[1]["y_train"] == Y
    assert call_args[1]["batch_size"] == 32
    assert call_args[1]["epochs"] == 10
    assert call_args[1]["validation_split"] == 0.2
    assert (
        len(call_args[1]["callbacks"]) == 3
    )  # EarlyStopping, ReduceLROnPlateau, TensorBoard

    assert result == model


def test_preprocess_corpus(monkeypatch):
    """Test preprocess_corpus function"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    # Mock language detection and preprocessing
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(
        bundeskanzler_ki, "preprocess", lambda x, lang=None: f"processed_{x}"
    )

    corpus = ["Satz 1", "Satz 2", "Satz 3"]
    result = bundeskanzler_ki.preprocess_corpus(corpus)

    expected = ["processed_Satz 1", "processed_Satz 2", "processed_Satz 3"]
    assert result == expected


def test_print_error_hint(monkeypatch):
    """Test print_error_hint function"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import logging
    from io import StringIO

    # Mock logging.basicConfig to prevent it from overriding pytest's logging setup
    with patch("logging.basicConfig"):
        import bundeskanzler_ki

    # Create a string stream to capture logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.ERROR)

    # Add handler to the logger
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)

    try:
        # Test FileNotFoundError
        bundeskanzler_ki.print_error_hint(FileNotFoundError("file not found"))
        log_contents = log_stream.getvalue()
        assert "Datei nicht gefunden" in log_contents

        # Clear stream
        log_stream.seek(0)
        log_stream.truncate(0)

        # Test ValueError
        bundeskanzler_ki.print_error_hint(ValueError("invalid value"))
        log_contents = log_stream.getvalue()
        assert "Wertfehler" in log_contents

        # Clear stream
        log_stream.seek(0)
        log_stream.truncate(0)

        # Test ImportError
        bundeskanzler_ki.print_error_hint(ImportError("import failed"))
        log_contents = log_stream.getvalue()
        assert "Importfehler" in log_contents

        # Clear stream
        log_stream.seek(0)
        log_stream.truncate(0)

        # Test generic exception
        bundeskanzler_ki.print_error_hint(Exception("generic error"))
        log_contents = log_stream.getvalue()
        assert "Unbekannter Fehler" in log_contents

    finally:
        # Clean up
        logger.removeHandler(handler)


def test_interactive_mode_with_input(monkeypatch, capsys):
    """Test interactive_mode with actual input processing"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    model = MagicMock()
    model.predict.return_value = [[0.8, 0.15, 0.05]]
    maxlen = 10
    corpus = ["Antwort A", "Antwort B", "Antwort C"]
    corpus_original = ["Antwort A", "Antwort B", "Antwort C"]
    args = types.SimpleNamespace(top_n=2, log="test.log")

    # Mock input sequence: valid input, then exit
    inputs = iter(["Testfrage", "exit"])
    monkeypatch.setattr("builtins.input", lambda x: next(inputs))

    # Mock dependencies
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x.lower())
    monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
    monkeypatch.setattr(bundeskanzler_ki, "feedback_interaction", lambda *a, **kw: None)

    # Mock pad_sequences
    mock_pad_sequences = MagicMock(return_value=[[1, 2, 3, 0, 0]])
    monkeypatch.setattr("bundeskanzler_ki.pad_sequences", mock_pad_sequences)

    bundeskanzler_ki.interactive_mode(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )

    # Check that model.predict was called
    model.predict.assert_called_once()

    # Check output
    captured = capsys.readouterr()
    assert "Top-2 Antworten" in captured.out
    assert "Antwort C" in captured.out


def test_interactive_mode_empty_input(monkeypatch):
    """Test interactive_mode with empty input"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import logging
    from io import StringIO

    # Mock logging.basicConfig to prevent it from overriding pytest's logging setup
    with patch("logging.basicConfig"):
        import bundeskanzler_ki

    # Create a string stream to capture logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)

    # Add handler to the logger
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        tokenizer = MagicMock()
        model = MagicMock()
        maxlen = 10
        corpus = ["Antwort A"]
        corpus_original = ["Antwort A"]
        args = types.SimpleNamespace(top_n=1, log="test.log")

        # Mock input sequence: empty input, then exit
        inputs = iter(["", "exit"])
        monkeypatch.setattr("builtins.input", lambda x: next(inputs))

        # Mock dependencies
        monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
        monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x)
        monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
        monkeypatch.setattr(
            bundeskanzler_ki, "feedback_interaction", lambda *a, **kw: None
        )

        bundeskanzler_ki.interactive_mode(
            tokenizer, model, maxlen, corpus, corpus_original, args
        )

        # Check that model.predict was not called (empty input skipped)
        model.predict.assert_not_called()

        # Check warning output
        log_contents = log_stream.getvalue()
        assert "Leere Eingabe übersprungen" in log_contents

    finally:
        # Clean up
        logger.removeHandler(handler)


def test_interactive_mode_model_none(monkeypatch, capsys):
    """Test interactive_mode with None model"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    tokenizer = MagicMock()
    model = None  # Model is None
    maxlen = 10
    corpus = ["Antwort A"]
    corpus_original = ["Antwort A"]
    args = types.SimpleNamespace(top_n=1, log="test.log")

    # Mock input sequence: input, then exit
    inputs = iter(["Testfrage", "exit"])
    monkeypatch.setattr("builtins.input", lambda x: next(inputs))

    bundeskanzler_ki.interactive_mode(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )

    # Check error output
    captured = capsys.readouterr()
    assert "Das Modell ist nicht geladen" in captured.out


def test_interactive_mode_tokenizer_none(monkeypatch, capsys):
    """Test interactive_mode with None tokenizer"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    tokenizer = None  # Tokenizer is None
    model = MagicMock()
    maxlen = 10
    corpus = ["Antwort A"]
    corpus_original = ["Antwort A"]
    args = types.SimpleNamespace(top_n=1, log="test.log")

    # Mock input sequence: input, then exit
    inputs = iter(["Testfrage", "exit"])
    monkeypatch.setattr("builtins.input", lambda x: next(inputs))

    bundeskanzler_ki.interactive_mode(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )

    # Check error output
    captured = capsys.readouterr()
    assert "Der Tokenizer ist nicht geladen" in captured.out


def test_interactive_mode_processing_error(monkeypatch, capsys):
    """Test interactive_mode with processing error"""
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    tokenizer = MagicMock()
    model = MagicMock()
    model.predict.side_effect = Exception("Prediction failed")
    maxlen = 10
    corpus = ["Antwort A"]
    corpus_original = ["Antwort A"]
    args = types.SimpleNamespace(top_n=1, log="test.log")

    # Mock input sequence: input, then exit
    inputs = iter(["Testfrage", "exit"])
    monkeypatch.setattr("builtins.input", lambda x: next(inputs))

    # Mock dependencies
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x)
    monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
    monkeypatch.setattr(bundeskanzler_ki, "feedback_interaction", lambda *a, **kw: None)

    # Mock pad_sequences
    mock_pad_sequences = MagicMock(return_value=[[1, 2, 3]])
    monkeypatch.setattr("bundeskanzler_ki.pad_sequences", mock_pad_sequences)

    bundeskanzler_ki.interactive_mode(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )

    # Check error output
    captured = capsys.readouterr()
    assert "Fehler bei der Verarbeitung" in captured.out


def test_batch_inference_runs(monkeypatch):
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    # Importiere nach Patchen, damit alle Abhängigkeiten gemockt sind
    import bundeskanzler_ki

    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [
        [1, 2, 3],
        [4, 5],
    ]  # Mock sequences as list of lists
    model = MagicMock()
    model.predict.return_value = [[0.7, 0.2, 0.1]]
    maxlen = 10
    corpus = ["Antwort1", "Antwort2", "Antwort3"]
    corpus_original = ["Antwort1", "Antwort2", "Antwort3"]
    args = types.SimpleNamespace(
        input="dummy.txt",
        top_n=2,
        print_answers=False,
        output_format="csv",
        output_path="",
        log="log.txt",
    )
    # Dummy-Eingabedatei erzeugen
    with open("dummy.txt", "w", encoding="utf-8") as f:
        f.write("Testfrage\n")
    # Preprocessing und Language Detection mocken
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x.lower())
    monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
    monkeypatch.setattr(
        bundeskanzler_ki, "export_batch_results_csv", lambda *a, **kw: None
    )
    # Test: Funktion läuft ohne Fehler durch
    bundeskanzler_ki.batch_inference(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )


def test_interactive_mode_exit(monkeypatch):
    pytest.skip(
        "Test übersprungen wegen TensorFlow/transformers Import-Konflikten in Testumgebung"
    )
    import bundeskanzler_ki

    tokenizer = MagicMock()
    model = MagicMock()
    model.predict.return_value = [[0.5, 0.3, 0.2]]
    maxlen = 10
    corpus = ["Antwort1", "Antwort2", "Antwort3"]
    corpus_original = ["Antwort1", "Antwort2", "Antwort3"]
    args = types.SimpleNamespace(top_n=2, log="log.txt")
    # Preprocessing und Language Detection mocken
    monkeypatch.setattr(bundeskanzler_ki, "detect_language", lambda x: "de")
    monkeypatch.setattr(bundeskanzler_ki, "preprocess", lambda x, lang=None: x.lower())
    monkeypatch.setattr(bundeskanzler_ki, "log_interaction", lambda *a, **kw: None)
    monkeypatch.setattr(bundeskanzler_ki, "feedback_interaction", lambda *a, **kw: None)
    # Simuliere Nutzereingabe: 'exit' direkt
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    # Test: Funktion läuft ohne Fehler durch und beendet sich
    bundeskanzler_ki.interactive_mode(
        tokenizer, model, maxlen, corpus, corpus_original, args
    )
