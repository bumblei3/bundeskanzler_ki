import pytest
from language_detection import (
    detect_language,
    get_supported_languages,
    is_language_supported,
)


def test_detect_language_de():
    text = "Die Bundesregierung beschließt neue Maßnahmen."
    lang = detect_language(text)
    assert lang == "de"


def test_detect_language_en():
    text = "The government decides on new measures."
    lang = detect_language(text)
    assert lang == "en"


def test_detect_language_unknown():
    text = "asdfghjklqwertzuiop"
    lang = detect_language(text)
    assert lang == "unknown"


def test_get_supported_languages():
    langs = get_supported_languages()
    assert "de" in langs and "en" in langs
    assert isinstance(langs, dict)


def test_is_language_supported():
    assert is_language_supported("de")
    assert not is_language_supported("ru")
