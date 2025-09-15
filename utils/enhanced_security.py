"""
Erweiterte Sicherheit für Bundeskanzler KI
Beinhaltet Input-Validierung, Rate-Limiting, Content-Filtering und Threat-Detection
"""

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EnhancedSecuritySystem:
    """
    Umfassendes Sicherheitssystem mit:
    - Input-Validierung und Sanitization
    - Rate-Limiting und Abuse-Detection
    - Content-Filtering für sensible Themen
    - Threat-Detection und Logging
    - Compliance-Monitoring
    """

    def __init__(
        self, max_requests_per_minute: int = 60, max_requests_per_hour: int = 1000
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour

        # Rate-Limiting Datenstrukturen
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips = set()
        self.suspicious_patterns = set()

        # Content-Filtering
        self.forbidden_words = self._load_forbidden_words()
        self.sensitive_topics = self._load_sensitive_topics()

        # Threat-Detection
        self.threat_patterns = self._load_threat_patterns()
        self.anomaly_scores = defaultdict(float)

        # Compliance-Logging
        self.compliance_log = deque(maxlen=5000)
        self.data_privacy_flags = set()

        # Konfiguration
        self.enable_content_filtering = True
        self.enable_rate_limiting = True
        self.enable_threat_detection = True

    def _load_forbidden_words(self) -> Set[str]:
        """Lädt Liste verbotener Wörter"""
        return {
            # Offensive Sprache
            "verdammt",
            "scheiße",
            "arsch",
            "wichser",
            "fotze",
            "schwanz",
            # Diskriminierung
            "nazi",
            "hitler",
            "juden",
            "neger",
            "schwuchtel",
            "kanake",
            # Illegale Aktivitäten
            "drogen",
            "kokain",
            "heroin",
            "meth",
            "ecstasy",
            "waffe",
            "bombe",
            "mord",
            "töten",
            "erschießen",
            "vergewaltigen",
            # Politisch sensible Themen
            "umsturz",
            "revolution",
            "putsch",
            "terror",
            "anschlag",
        }

    def _load_sensitive_topics(self) -> Dict[str, List[str]]:
        """Lädt sensible Themen und zugehörige Keywords"""
        return {
            "datenschutz": ["personendaten", "gdpr", "dsgvo", "privatsphäre"],
            "geheimdienste": [
                "bundesnachrichtendienst",
                "verfassungsschutz",
                "nsa",
                "cia",
            ],
            "militär": ["bundeswehr", "atomwaffe", "nuklear", "krieg"],
            "gesundheit": ["corona", "pandemie", "impfung", "nebenwirkung"],
            "politik_extrem": [
                "rechtsradikal",
                "linksextrem",
                "extremismus",
                "radikalisierung",
            ],
        }

    def _load_threat_patterns(self) -> List[Dict[str, Any]]:
        """Lädt Muster für Threat-Detection"""
        return [
            {
                "name": "prompt_injection",
                "pattern": r"(?i)(ignore|override|forget|disregard).*(instruction|previous|above)",
                "severity": "high",
                "action": "block",
            },
            {
                "name": "jailbreak_attempt",
                "pattern": r"(?i)(developer|admin|root|system).*(mode|access|prompt)",
                "severity": "high",
                "action": "block",
            },
            {
                "name": "malicious_code",
                "pattern": r"(?i)(eval|exec|system|subprocess|os\.|import\s+os)",
                "severity": "critical",
                "action": "block",
            },
            {
                "name": "data_exfiltration",
                "pattern": r"(?i)(dump|export|send|transmit).*(data|database|user|password)",
                "severity": "high",
                "action": "flag",
            },
            {
                "name": "spam_pattern",
                "pattern": r"(.)\1{10,}",  # Wiederholte Zeichen
                "severity": "low",
                "action": "flag",
            },
        ]

    def validate_input(
        self, input_text: str, user_id: str = "anonymous", ip_address: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Führt umfassende Input-Validierung durch

        Returns:
            Dict mit validation_result, flags, und recommendations
        """
        result = {
            "is_valid": True,
            "flags": [],
            "warnings": [],
            "recommendations": [],
            "risk_score": 0.0,
            "sanitized_input": input_text,
        }

        # Längen-Validierung
        if len(input_text) > 2000:
            result["flags"].append("input_too_long")
            result["warnings"].append("Eingabe ist sehr lang (>2000 Zeichen)")
            result["risk_score"] += 0.3

        if len(input_text) < 3:
            result["flags"].append("input_too_short")
            result["warnings"].append("Eingabe ist sehr kurz (<3 Zeichen)")
            result["is_valid"] = False

        # Rate-Limiting prüfen
        if self.enable_rate_limiting:
            rate_limit_result = self.check_rate_limit(user_id, ip_address)
            if not rate_limit_result["allowed"]:
                result["is_valid"] = False
                result["flags"].append("rate_limit_exceeded")
                result["recommendations"].append(
                    "Bitte warten Sie einen Moment vor der nächsten Anfrage"
                )

        # Content-Filtering
        if self.enable_content_filtering:
            content_result = self.filter_content(input_text)
            result["flags"].extend(content_result["flags"])
            result["warnings"].extend(content_result["warnings"])
            result["risk_score"] += content_result["risk_score"]

            if content_result["blocked"]:
                result["is_valid"] = False

        # Threat-Detection
        if self.enable_threat_detection:
            threat_result = self.detect_threats(input_text, user_id)
            result["flags"].extend(threat_result["flags"])
            result["risk_score"] += threat_result["risk_score"]

            if threat_result["blocked"]:
                result["is_valid"] = False

        # Input-Sanitization
        result["sanitized_input"] = self.sanitize_input(input_text)

        # Compliance-Logging
        self.log_compliance_event(
            {
                "event_type": "input_validation",
                "user_id": user_id,
                "ip_address": ip_address,
                "input_length": len(input_text),
                "validation_result": result["is_valid"],
                "risk_score": result["risk_score"],
                "flags": result["flags"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Kritische Flags führen zu Blockierung
        critical_flags = ["threat_detected", "forbidden_content", "rate_limit_exceeded"]
        if any(flag in result["flags"] for flag in critical_flags):
            result["is_valid"] = False

        return result

    def check_rate_limit(self, user_id: str, ip_address: str) -> Dict[str, Any]:
        """Prüft Rate-Limiting für User/IP"""
        current_time = time.time()

        # Kombiniere User-ID und IP für bessere Identifikation
        identifier = f"{user_id}_{ip_address}"

        # Bereinige alte Einträge (> 1 Stunde)
        while (
            self.request_history[identifier]
            and current_time - self.request_history[identifier][0] > 3600
        ):
            self.request_history[identifier].popleft()

        # Zähle Requests in letzten 60 Sekunden
        minute_ago = current_time - 60
        recent_requests = sum(
            1 for t in self.request_history[identifier] if t > minute_ago
        )

        # Zähle Requests in letzten 3600 Sekunden
        hour_ago = current_time - 3600
        hourly_requests = sum(
            1 for t in self.request_history[identifier] if t > hour_ago
        )

        allowed = (
            recent_requests < self.max_requests_per_minute
            and hourly_requests < self.max_requests_per_hour
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {identifier}: {recent_requests}/min, {hourly_requests}/hour"
            )

        # Füge aktuellen Request hinzu (auch wenn blockiert)
        self.request_history[identifier].append(current_time)

        return {
            "allowed": allowed,
            "requests_per_minute": recent_requests,
            "requests_per_hour": hourly_requests,
            "limit_per_minute": self.max_requests_per_minute,
            "limit_per_hour": self.max_requests_per_hour,
        }

    def filter_content(self, input_text: str) -> Dict[str, Any]:
        """Filtert Inhalt auf verbotene Wörter und sensible Themen"""
        result = {"blocked": False, "flags": [], "warnings": [], "risk_score": 0.0}

        text_lower = input_text.lower()

        # Prüfe verbotene Wörter
        found_forbidden = []
        for word in self.forbidden_words:
            if word in text_lower:
                found_forbidden.append(word)
                result["risk_score"] += 0.5

        if found_forbidden:
            result["flags"].append("forbidden_words")
            result["warnings"].append(
                f"Verbotene Wörter gefunden: {', '.join(found_forbidden[:3])}"
            )
            if len(found_forbidden) > 2:
                result["blocked"] = True

        # Prüfe sensible Themen
        for topic, keywords in self.sensitive_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                result["flags"].append(f"sensitive_topic_{topic}")
                result["warnings"].append(f"Sensibles Thema erkannt: {topic}")
                result["risk_score"] += 0.2

        return result

    def detect_threats(self, input_text: str, user_id: str) -> Dict[str, Any]:
        """Erkennt Bedrohungen und verdächtige Patterns"""
        result = {"blocked": False, "flags": [], "risk_score": 0.0}

        for pattern_info in self.threat_patterns:
            pattern = re.compile(pattern_info["pattern"])
            if pattern.search(input_text):
                result["flags"].append(f"threat_{pattern_info['name']}")
                result["risk_score"] += (
                    0.8 if pattern_info["severity"] == "high" else 0.4
                )

                if pattern_info["action"] == "block":
                    result["blocked"] = True

                logger.warning(
                    f"Threat detected: {pattern_info['name']} from user {user_id}"
                )

        # Anomaly-Score für User aktualisieren
        self.anomaly_scores[user_id] += result["risk_score"]

        # Hoher Anomaly-Score führt zu temporärer Blockierung
        if self.anomaly_scores[user_id] > 3.0:
            result["blocked"] = True
            result["flags"].append("high_anomaly_score")
            logger.warning(
                f"High anomaly score for user {user_id}: {self.anomaly_scores[user_id]}"
            )

        return result

    def sanitize_input(self, input_text: str) -> str:
        """Sanitisiert User-Input"""
        # Entferne übermäßige Leerzeichen
        sanitized = re.sub(r"\s+", " ", input_text.strip())

        # Entferne potenziell gefährliche Zeichen (aber erlaube deutsche Umlaute)
        sanitized = re.sub(r"[^\w\säöüÄÖÜß.,!?-]", "", sanitized)

        # Begrenze Länge
        if len(sanitized) > 1000:
            sanitized = sanitized[:997] + "..."

        return sanitized

    def log_compliance_event(self, event: Dict[str, Any]):
        """Loggt Compliance-relevante Events"""
        self.compliance_log.append(event)

        # Schreibe kritische Events sofort in Log
        if event.get("risk_score", 0) > 0.5 or not event.get("validation_result", True):
            logger.info(f"Compliance Event: {event}")

    def get_security_report(self) -> Dict[str, Any]:
        """Gibt einen Sicherheits-Report zurück"""
        recent_events = list(self.compliance_log)[-100:]  # Letzte 100 Events

        blocked_requests = sum(
            1 for e in recent_events if not e.get("validation_result", True)
        )
        high_risk_requests = sum(
            1 for e in recent_events if e.get("risk_score", 0) > 0.5
        )

        return {
            "total_requests": len(recent_events),
            "blocked_requests": blocked_requests,
            "high_risk_requests": high_risk_requests,
            "block_rate": blocked_requests / max(len(recent_events), 1),
            "active_threats": len([u for u in self.anomaly_scores.values() if u > 1.0]),
            "rate_limited_users": len(
                [h for h in self.request_history.values() if len(h) > 50]
            ),
        }

    def reset_user_anomaly_score(self, user_id: str):
        """Setzt Anomaly-Score für einen User zurück"""
        if user_id in self.anomaly_scores:
            del self.anomaly_scores[user_id]

    def add_to_blacklist(self, identifier: str, reason: str = "manual"):
        """Fügt einen Identifier zur Blacklist hinzu"""
        self.blocked_ips.add(identifier)
        logger.warning(f"Added to blacklist: {identifier} (reason: {reason})")

    def remove_from_blacklist(self, identifier: str):
        """Entfernt einen Identifier von der Blacklist"""
        self.blocked_ips.discard(identifier)
        logger.info(f"Removed from blacklist: {identifier}")

    def export_security_report(self, file_path: str):
        """Exportiert detaillierten Sicherheits-Report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "security_metrics": self.get_security_report(),
            "recent_compliance_events": list(self.compliance_log)[-50:],
            "blacklisted_identifiers": list(self.blocked_ips),
            "high_anomaly_users": {
                k: v for k, v in self.anomaly_scores.items() if v > 1.0
            },
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Security report exported to {file_path}")
