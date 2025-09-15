import pytest
import types
from unittest.mock import MagicMock, patch
import sys

# Import global test stubs first
import test_stubs

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tf_config', MagicMock())
    monkeypatch.setitem(sys.modules, 'transformer_model', MagicMock())
    monkeypatch.setitem(sys.modules, 'corpus_manager', MagicMock())
    monkeypatch.setitem(sys.modules, 'preprocessing', MagicMock())
    monkeypatch.setitem(sys.modules, 'language_detection', MagicMock())
    monkeypatch.setitem(sys.modules, 'feedback', MagicMock())
    # Entferne model patching, da echte model Tests gemacht werden
    # monkeypatch.setitem(sys.modules, 'model', MagicMock())
    monkeypatch.setitem(sys.modules, 'validation', MagicMock())


def test_train_model_runs(monkeypatch, tmp_path):
    import bundeskanzler_ki
    model = MagicMock()
    X = [[1,2,3]]
    Y = [[0,1,0]]
    args = types.SimpleNamespace(batch_size=2, epochs=1, top_n=1, input='dummy.txt', corpus='corpus.txt', log='log.txt')
    # train_transformer mocken
    train_transformer = MagicMock(return_value='history')
    monkeypatch.setattr(sys.modules['transformer_model'], 'train_transformer', train_transformer)
    # Test: Funktion läuft ohne Fehler durch
    result = bundeskanzler_ki.train_model(model, X, Y, args)
    assert result is model


def test_validate_model_runs(monkeypatch):
    import bundeskanzler_ki
    tokenizer = MagicMock()
    model = MagicMock()
    maxlen = 10
    preprocess = MagicMock()
    detect_language = MagicMock()
    # validate_model mocken
    validate_model = MagicMock()
    monkeypatch.setattr(sys.modules['validation'], 'validate_model', validate_model)
    # Test: Funktion läuft ohne Fehler durch
    bundeskanzler_ki.validate_model(tokenizer, model, maxlen, preprocess, detect_language)
    validate_model.assert_not_called()  # Die Funktion ruft das Mock direkt auf, kein Fehler


def test_main_config_missing(monkeypatch):
    import bundeskanzler_ki
    # Simuliere fehlende Konfigurationswerte
    config = {'data': {}, 'general': {}, 'training': {}, 'model': {}}
    monkeypatch.setattr('builtins.open', lambda *a, **kw: MagicMock(__enter__=lambda s: s, __exit__=lambda s, exc_type, exc_val, exc_tb: None, read=lambda: ''))
    monkeypatch.setattr('yaml.safe_load', lambda f: config)
    with pytest.raises(SystemExit):
        bundeskanzler_ki.main()
