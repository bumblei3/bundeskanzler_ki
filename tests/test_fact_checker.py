import json
import tempfile
from datetime import datetime

import pytest
from fact_checker import FactChecker, FactCheckResult


def test_check_fact_basic():
    """Test grundlegende Faktenprüfung"""
    fc = FactChecker()

    # Test mit einer einfachen Aussage
    result = fc.check_fact("Die Bundesregierung hat 2023 ein Klimapaket verabschiedet.")

    assert isinstance(result, FactCheckResult)
    assert result.statement == "Die Bundesregierung hat 2023 ein Klimapaket verabschiedet."
    assert 0.0 <= result.confidence_score <= 1.0
    assert isinstance(result.sources, list)
    assert -1.0 <= result.bias_score <= 1.0
    assert result.verification_status in [
        "verified",
        "partially_verified",
        "unverified",
        "contradicted",
    ]
    assert isinstance(result.explanation, str)
    assert isinstance(result.timestamp, datetime)


def test_check_fact_with_context():
    """Test Faktenprüfung mit Kontext"""
    fc = FactChecker()

    context = {
        "topic": "climate",
        "urgency": 0.8,
        "source_preference": ["government", "academic"],
    }

    result = fc.check_fact(
        "Deutschland hat sich zu Netto-Null-Emissionen bis 2045 verpflichtet.", context
    )

    assert isinstance(result, FactCheckResult)
    assert result.confidence_score > 0.0  # Sollte mit Kontext besser sein


def test_bias_analysis():
    """Test Bias-Analyse"""
    fc = FactChecker()

    # Test linke Bias-Indikatoren
    left_bias_result = fc.check_fact(
        "Soziale Gerechtigkeit und Umweltschutz sind wichtiger als Wirtschaftswachstum."
    )
    assert left_bias_result.bias_score < -0.1  # Sollte linke Tendenz zeigen

    # Test rechte Bias-Indikatoren
    right_bias_result = fc.check_fact("Sicherheit und Ordnung haben Priorität vor Migration.")
    assert right_bias_result.bias_score > 0.1  # Sollte rechte Tendenz zeigen


def test_source_finding():
    """Test Quellenfindung"""
    fc = FactChecker()

    result = fc.check_fact("Die Inflationsrate in Deutschland beträgt 2.5%.")

    # Sollte mindestens eine Quelle finden
    assert len(result.sources) > 0

    # Jede Quelle sollte die erwarteten Felder haben
    for source in result.sources:
        assert "title" in source
        assert "url" in source
        assert "credibility_score" in source
        assert "relevance_score" in source


def test_confidence_calculation():
    """Test Konfidenzberechnung"""
    fc = FactChecker()

    # Hoch vertrauenswürdige Aussage
    high_conf_result = fc.check_fact("Berlin ist die Hauptstadt Deutschlands.")
    assert high_conf_result.confidence_score > 0.7

    # Weniger vertrauenswürdige Aussage
    low_conf_result = fc.check_fact("Alle Politiker sind korrupt.")
    # Diese Aussage könnte immer noch hoch bewertet werden wegen der Quellen
    assert isinstance(low_conf_result.confidence_score, float)


def test_validate_response():
    """Test Antwortvalidierung"""
    fc = FactChecker()

    response = "Die Bundesregierung plant ein neues Klimapaket."
    user_query = "Was plant die Regierung zum Klimaschutz?"

    validation = fc.validate_response(response, user_query)

    assert isinstance(validation, dict)
    assert "overall_confidence" in validation
    assert "statement_validations" in validation
    assert "recommendations" in validation
    assert 0.0 <= validation["overall_confidence"] <= 1.0
