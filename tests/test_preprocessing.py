import pytest
from preprocessing import detect_lang, preprocess


def test_preprocess_deutsch():
    text = "Die Bundesregierung arbeitet für den Klimaschutz!"
    result = preprocess(text, lang="de")
    # Erwartet: Kleinbuchstaben, keine Satzzeichen, keine Stopwords, gestemmt
    assert "regierung" in result or "regier" in result
    assert "klimaschutz" in result or "klimasch" in result
    assert "die" not in result
    assert "!" not in result


def test_preprocess_englisch():
    text = "The government is working for climate protection!"
    result = preprocess(text, lang="en")
    # Erwartet: Kleinbuchstaben, keine Satzzeichen, keine Stopwords, lemmatisiert
    assert "government" in result
    assert "climate" in result
    assert "protection" in result
    assert "the" not in result
    assert "!" not in result


def test_detect_lang_de():
    text = "Die Bundesregierung beschließt neue Maßnahmen."
    assert detect_lang(text) == "de"


def test_detect_lang_en():
    text = "The government decides on new measures."
    assert detect_lang(text) == "en"
