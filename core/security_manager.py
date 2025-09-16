#!/usr/bin/env python3
"""
🛡️ Sicherheit & Ethik Manager für Bundeskanzler-KI
===============================================

Umfassendes Sicherheitsmodul mit:
- Input-Validation und Content-Filtering
- Bias-Detection und Fairness-Monitoring
- Abuse-Detection und Missbrauchserkennung
- Ethics-Reporting und Transparenz
- Security-Logging und Incident-Management

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Konfiguriere Security-Logging
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)
security_handler = logging.FileHandler("logs/security.log")
security_handler.setFormatter(
    logging.Formatter("%(asctime)s - SECURITY - %(levelname)s - %(message)s")
)
security_logger.addHandler(security_handler)


class SecurityManager:
    """
    🛡️ Umfassender Sicherheit & Ethik Manager
    """

    def __init__(self):
        """Initialisiert den Security Manager"""
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)

        # Sicherheitskonfiguration laden
        self.security_config = self._load_security_config()

        # Bias-Detection Patterns
        self.bias_patterns = self._load_bias_patterns()

        # Abuse-Detection Patterns
        self.abuse_patterns = self._load_abuse_patterns()

        # Ethics-Monitoring
        self.ethics_log = []

        # Security-Stats
        self.security_stats = {
            "inputs_filtered": 0,
            "bias_detected": 0,
            "abuse_attempts": 0,
            "ethics_violations": 0,
            "last_incident": None,
        }

        security_logger.info("🛡️ Security Manager initialisiert")

    def _load_security_config(self) -> Dict[str, Any]:
        """Lädt Sicherheitskonfiguration"""
        config_file = self.config_dir / "security_config.json"

        if not config_file.exists():
            # Standard-Konfiguration erstellen
            default_config = {
                "input_validation": {
                    "max_length": 1000,
                    "min_length": 3,
                    "allowed_chars": "a-zA-ZäöüßÄÖÜ0-9 .,;:!?-()[]{}'\"",
                    "block_keywords": ["hack", "exploit", "attack", "virus", "malware"],
                },
                "content_filtering": {
                    "political_bias_threshold": 0.7,
                    "toxicity_threshold": 0.6,
                    "fact_check_required": True,
                },
                "abuse_detection": {
                    "max_requests_per_minute": 10,
                    "max_requests_per_hour": 50,
                    "block_duration_minutes": 15,
                },
                "ethics_reporting": {
                    "transparency_level": "high",
                    "bias_monitoring": True,
                    "source_verification": True,
                },
            }

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)

            return default_config

        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_bias_patterns(self) -> List[Dict[str, Any]]:
        """Lädt Bias-Detection Patterns"""
        return [
            {
                "pattern": r"\b(immer|nur|alle|keiner|niemand)\b.*\b(Deutsch|Migranten|Politiker)\b",
                "bias_type": "generalization",
                "severity": "medium",
                "description": "Verallgemeinernde Aussagen über Gruppen",
            },
            {
                "pattern": r"\b(links|rechts|liberal|konservativ)\b.*\b(immer|nie|nur)\b",
                "bias_type": "political_stereotyping",
                "severity": "high",
                "description": "Politische Stereotypisierung",
            },
            {
                "pattern": r"\b(richtig|falsch|gut|schlecht)\b.*\b(Partei|Politiker)\b",
                "bias_type": "value_judgment",
                "severity": "low",
                "description": "Werturteile über politische Akteure",
            },
        ]

    def _load_abuse_patterns(self) -> List[Dict[str, Any]]:
        """Lädt Abuse-Detection Patterns"""
        return [
            {
                "pattern": r"(?i)(hack|exploit|attack|virus|malware|trojan)",
                "abuse_type": "malicious_intent",
                "severity": "high",
                "description": "Versuch schädlicher Aktivitäten",
            },
            {
                "pattern": r"(?i)(bomb|weapon|violence|terror)",
                "abuse_type": "violent_content",
                "severity": "critical",
                "description": "Gewaltbezogene Inhalte",
            },
            {
                "pattern": r"(?i)(password|credential|secret|key)",
                "abuse_type": "credential_theft",
                "severity": "high",
                "description": "Versuch von Credential-Diebstahl",
            },
            {
                "pattern": r"(?i)(spam|advertisement|marketing)",
                "abuse_type": "spam",
                "severity": "low",
                "description": "Spam oder unerwünschte Werbung",
            },
        ]

    def validate_input(
        self, input_text: str, user_id: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validiert Eingabe auf Sicherheit und Angemessenheit

        Args:
            input_text: Zu validierender Text
            user_id: Optionale User-ID für Rate-Limiting

        Returns:
            (is_valid, reason, metadata)
        """
        metadata = {
            "validation_time": datetime.now().isoformat(),
            "input_length": len(input_text),
            "user_id": user_id or "anonymous",
        }

        # Längen-Validierung
        config = self.security_config["input_validation"]
        if len(input_text) < config["min_length"]:
            return False, "Eingabe zu kurz", metadata

        if len(input_text) > config["max_length"]:
            return False, "Eingabe zu lang", metadata

        # Zeichen-Validierung
        allowed_pattern = f"[{re.escape(config['allowed_chars'])}]"
        if not re.search(allowed_pattern, input_text):
            return False, "Unzulässige Zeichen enthalten", metadata

        # Keyword-Blockierung
        for keyword in config["block_keywords"]:
            if keyword.lower() in input_text.lower():
                security_logger.warning(f"🚫 Blocked keyword '{keyword}' in input from {user_id}")
                self.security_stats["inputs_filtered"] += 1
                return False, f"Inhalt blockiert: {keyword}", metadata

        # Abuse-Detection
        abuse_result = self._detect_abuse(input_text)
        if abuse_result["detected"]:
            security_logger.warning(f"🚨 Abuse detected: {abuse_result['type']} from {user_id}")
            self.security_stats["abuse_attempts"] += 1
            self.security_stats["last_incident"] = datetime.now().isoformat()
            return False, f"Missbrauch erkannt: {abuse_result['type']}", metadata

        return True, "Eingabe validiert", metadata

    def _detect_abuse(self, text: str) -> Dict[str, Any]:
        """Erkennt Missbrauch in Text"""
        for pattern_info in self.abuse_patterns:
            if re.search(pattern_info["pattern"], text, re.IGNORECASE):
                return {
                    "detected": True,
                    "type": pattern_info["abuse_type"],
                    "severity": pattern_info["severity"],
                    "description": pattern_info["description"],
                }

        return {"detected": False}

    def detect_bias(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Erkennt Bias und Voreingenommenheit in Text

        Args:
            text: Zu analysierender Text
            context: Optionaler Kontext (z.B. Quelle, Thema)

        Returns:
            Bias-Analyse Ergebnis
        """
        bias_detected = []
        bias_score = 0.0

        for pattern_info in self.bias_patterns:
            matches = re.findall(pattern_info["pattern"], text, re.IGNORECASE)
            if matches:
                bias_detected.append(
                    {
                        "type": pattern_info["bias_type"],
                        "severity": pattern_info["severity"],
                        "matches": matches,
                        "description": pattern_info["description"],
                    }
                )

                # Severity zu Score konvertieren
                severity_scores = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 1.0}
                bias_score = max(bias_score, severity_scores.get(pattern_info["severity"], 0.5))

        result = {
            "bias_detected": len(bias_detected) > 0,
            "bias_score": bias_score,
            "bias_types": bias_detected,
            "recommendations": [],
        }

        if result["bias_detected"]:
            self.security_stats["bias_detected"] += 1
            security_logger.info(f"⚠️ Bias detected: score {bias_score:.2f}")

            # Empfehlungen generieren
            if bias_score >= 0.8:
                result["recommendations"].append(
                    "Hohe Bias-Wahrscheinlichkeit - Antwort überprüfen"
                )
            elif bias_score >= 0.6:
                result["recommendations"].append(
                    "Mittlere Bias-Wahrscheinlichkeit - Quelle verifizieren"
                )

        return result

    def filter_content(
        self, content: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Filtert Inhalt auf Grundlage von Sicherheitsrichtlinien

        Args:
            content: Zu filternder Inhalt
            metadata: Metadaten des Inhalts

        Returns:
            (is_allowed, reason, filtered_metadata)
        """
        # Bias-Analyse
        bias_analysis = self.detect_bias(content, metadata)

        # Content-Filtering basierend auf Konfiguration
        config = self.security_config["content_filtering"]

        if bias_analysis["bias_score"] >= config["political_bias_threshold"]:
            return (
                False,
                f"Bias-Score zu hoch: {bias_analysis['bias_score']:.2f}",
                {"bias_analysis": bias_analysis, "filter_reason": "political_bias"},
            )

        # Source-Verification (falls erforderlich)
        if config["fact_check_required"] and not metadata.get("verified", False):
            return False, "Quelle nicht verifiziert", {"filter_reason": "unverified_source"}

        return (
            True,
            "Inhalt freigegeben",
            {"bias_analysis": bias_analysis, "content_score": 1.0 - bias_analysis["bias_score"]},
        )

    def generate_ethics_report(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert Ethik-Report für eine Interaktion

        Args:
            interaction: Interaktionsdaten

        Returns:
            Ethik-Report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "interaction_id": hashlib.md5(
                f"{interaction.get('user_id', 'anonymous')}_{interaction.get('timestamp', datetime.now().isoformat())}".encode()
            ).hexdigest()[:8],
            "transparency_score": self._calculate_transparency_score(interaction),
            "fairness_score": self._calculate_fairness_score(interaction),
            "accountability_score": self._calculate_accountability_score(interaction),
            "issues": [],
            "recommendations": [],
        }

        # Ethik-Prüfungen
        if interaction.get("bias_detected", False):
            report["issues"].append("Bias in Antwort erkannt")
            report["recommendations"].append("Antwort auf Neutralität überprüfen")

        if not interaction.get("source_verified", False):
            report["issues"].append("Quelle nicht verifiziert")
            report["recommendations"].append("Informationsquelle validieren")

        if interaction.get("confidence", 1.0) < 0.5:
            report["issues"].append("Niedrige Antwort-Konfidenz")
            report["recommendations"].append("Zusätzliche Quellen konsultieren")

        # Report in Log speichern
        self.ethics_log.append(report)

        return report

    def _calculate_transparency_score(self, interaction: Dict[str, Any]) -> float:
        """Berechnet Transparenz-Score"""
        score = 0.0

        if interaction.get("sources"):
            score += 0.3
        if interaction.get("confidence") is not None:
            score += 0.3
        if interaction.get("explanation"):
            score += 0.4

        return min(score, 1.0)

    def _calculate_fairness_score(self, interaction: Dict[str, Any]) -> float:
        """Berechnet Fairness-Score"""
        score = 1.0

        if interaction.get("bias_detected", False):
            score -= 0.5
        if not interaction.get("multiple_sources", False):
            score -= 0.2

        return max(score, 0.0)

    def _calculate_accountability_score(self, interaction: Dict[str, Any]) -> float:
        """Berechnet Accountability-Score"""
        score = 0.0

        if interaction.get("user_id"):
            score += 0.3
        if interaction.get("timestamp"):
            score += 0.3
        if interaction.get("logged", False):
            score += 0.4

        return min(score, 1.0)

    def get_security_stats(self) -> Dict[str, Any]:
        """Gibt Sicherheitsstatistiken zurück"""
        return {
            **self.security_stats,
            "ethics_reports_generated": len(self.ethics_log),
            "active_config": self.security_config,
        }

    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "info"):
        """Loggt Sicherheitsereignis"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "severity": severity,
            "details": details,
        }

        if severity == "warning":
            security_logger.warning(f"Security Event: {event_type} - {details}")
        elif severity == "error":
            security_logger.error(f"Security Event: {event_type} - {details}")
        else:
            security_logger.info(f"Security Event: {event_type} - {details}")

        # Stats aktualisieren
        if event_type == "abuse_attempt":
            self.security_stats["abuse_attempts"] += 1
        elif event_type == "bias_detected":
            self.security_stats["bias_detected"] += 1
        elif event_type == "input_filtered":
            self.security_stats["inputs_filtered"] += 1

        self.security_stats["last_incident"] = datetime.now().isoformat()


# Singleton-Instanz
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Gibt Singleton-Instanz des Security Managers zurück"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# Test-Funktion
def test_security_features():
    """Testet Sicherheitsfeatures"""
    security = get_security_manager()

    # Test-Inputs
    test_cases = [
        ("Normale Frage: Was ist Klimapolitik?", True, "Sollte erlaubt sein"),
        ("Hack Versuch: Wie hacke ich das System?", False, "Sollte blockiert werden"),
        ("Spam: Kaufen Sie Bitcoin!", False, "Sollte als Spam erkannt werden"),
        ("Bias: Alle Politiker sind korrupt", True, "Bias wird erkannt aber nicht blockiert"),
    ]

    print("🧪 Sicherheitstests:")
    print("=" * 50)

    for test_input, expected_valid, description in test_cases:
        is_valid, reason, metadata = security.validate_input(test_input)

        status = "✅" if is_valid == expected_valid else "❌"
        print(f"{status} {description}")
        print(f"   Input: {test_input[:50]}...")
        print(f"   Valid: {is_valid}, Reason: {reason}")
        print()


if __name__ == "__main__":
    test_security_features()
