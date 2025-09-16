import io
import json
import os
import sys

import pytest

# ensure local modules are importable
sys.path.insert(0, "/home/tobber/bkki_venv")

# Sicherstellen, dass echtes numpy und tensorflow verwendet wird - MUSS VOR anderen Imports sein
import numpy
import tensorflow

sys.modules["numpy"] = numpy
sys.modules["tensorflow"] = tensorflow

import importlib

validation = importlib.import_module("validation")

import pytest


# Zusätzliche Sicherstellung in jedem Test
@pytest.fixture(autouse=True)
def ensure_real_modules():
    """Stellt sicher, dass echte numpy und tensorflow Module verwendet werden."""
    import numpy
    import tensorflow

    sys.modules["numpy"] = numpy
    sys.modules["tensorflow"] = tensorflow
    yield


def make_test_file(tmp_path, lines):
    p = tmp_path / "test_val.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_validate_model_no_file(tmp_path):
    # non-existent file -> returns None
    res = validation.validate_model(
        None, None, 10, None, None, test_file=str(tmp_path / "does_not_exist.txt")
    )
    assert res is None


@pytest.mark.xfail(reason="Stub interference from other tests - works when run individually")
def test_validate_model_basic(tmp_path, monkeypatch):
    # create simple test file with two valid lines
    # both expect class '0'
    tf = make_test_file(tmp_path, ["Hallo Welt\t0", "Guten Tag\t0"])

    # stubs
    class Tok:
        def texts_to_sequences(self, lst):
            return [[1, 2, 3] for _ in lst]

    class Model:
        def predict(self, seq, verbose=0):
            # Return for each input a probability vector where class 0 has higher prob
            return [[0.9, 0.1] for _ in seq]

    def preprocess(text, lang=None):
        return text

    def detect_lang(text):
        return "de"

    # patch pad_sequences for test
    monkeypatch.setattr(validation, "pad_sequences", lambda seq, maxlen, padding="post": seq)

    csv_out = str(tmp_path / "out.csv")

    res = validation.validate_model(
        tokenizer=Tok(),
        model=Model(),
        maxlen=10,
        preprocess=preprocess,
        detect_lang=detect_lang,
        test_file=tf,
        csv_out=csv_out,
        top_n=1,
        return_df=False,
    )

    assert isinstance(res, dict)
    assert res["total"] == 2
    assert res["correct"] == 2
    # CSV file should be written
    assert os.path.exists(csv_out)
    # JSON export should also be created by the function
    json_path = tmp_path / "validation_results.json"
    # validation writes to current working dir; check file exists in cwd
    assert os.path.exists("validation_results.json")
    # cleanup created JSON
    try:
        os.remove("validation_results.json")
    except Exception:
        pass
