#!/usr/bin/env python3
"""
Erweitertes Sicherheitssystem für die Bundeskanzler KI
Umfassender Schutz vor Angriffen, Content-Filtering und Audit-Trails
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import logging
import json
import re
from collections import defaultdict, deque
import threading
from queue import Queue
import ipaddress
import geoip2.database
import os

logger = logging.getLogger(__name__)

class AdvancedSecuritySystem:
    """
    Umfassendes Sicherheitssystem mit mehreren Schutzschichten
    """

    def __init__(self):
        # API-Schlüssel-Management
        self.api_keys = self._load_api_keys()
        self.key_usage = defaultdict(int)
        self.key_last_used = {}

        # Rate Limiting
        self.rate_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "concurrent_requests": 10
        }
        self.request_history = defaultdict(deque)
        self.active_requests = defaultdict(int)

        # Content-Filtering
        self.content_filters = self._initialize_content_filters()

        # Threat Detection
        self.threat_patterns = self._load_threat_patterns()
        self.suspicious_activities = deque(maxlen=1000)
        self.blocked_ips = set()
        self.blocked_users = set()

        # Audit Logging
        self.audit_log = deque(maxlen=10000)
        self.audit_queue = Queue()

        # GeoIP für Standort-basierte Sicherheit
        self.geoip_db = None
        self._load_geoip_database()

        # Intrusion Detection
        self.failed_login_attempts = defaultdict(list)
        self.brute_force_threshold = 5  # Anzahl fehlgeschlagener Versuche

        # Starte Sicherheits-Threads
        self._start_security_threads()

    def _start_security_threads(self):
        """Startet Sicherheits-Überwachungs-Threads"""
        audit_thread = threading.Thread(target=self._audit_logging_loop, daemon=True)
        audit_thread.start()

        monitoring_thread = threading.Thread(target=self._security_monitoring_loop, daemon=True)
        monitoring_thread.start()

        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()

        logger.info("✅ Erweitertes Sicherheitssystem aktiviert")

    def authenticate_request(self, api_key: str, request_data: Dict) -> Tuple[bool, str]:
        """
        Authentifiziert eine API-Anfrage

        Args:
            api_key: Der API-Schlüssel
            request_data: Anfrage-Daten

        Returns:
            Tuple[bool, str]: (Erfolgreich, Nachricht)
        """
        # Prüfe API-Schlüssel
        if api_key not in self.api_keys:
            self._log_security_event("invalid_api_key", {"key_hash": self._hash_string(api_key)})
            return False, "Ungültiger API-Schlüssel"

        # Prüfe API-Schlüssel-Status
        key_info = self.api_keys[api_key]
        if not key_info.get("active", True):
            return False, "API-Schlüssel ist deaktiviert"

        # Prüfe Ablaufdatum
        if "expires" in key_info:
            if datetime.now() > datetime.fromisoformat(key_info["expires"]):
                return False, "API-Schlüssel ist abgelaufen"

        # Rate Limiting prüfen
        client_ip = request_data.get("client_ip", "unknown")
        if not self._check_rate_limits(client_ip, api_key):
            self._log_security_event("rate_limit_exceeded", {"ip": client_ip, "key": self._hash_string(api_key)})
            return False, "Rate Limit überschritten"

        # Content-Filtering
        if not self._check_content_safety(request_data):
            self._log_security_event("content_filter_triggered", {
                "ip": client_ip,
                "content_hash": self._hash_string(str(request_data))
            })
            return False, "Inhalt verstößt gegen Sicherheitsrichtlinien"

        # Threat Detection
        if self._detect_threats(request_data):
            self._log_security_event("threat_detected", {"ip": client_ip})
            return False, "Potenzielle Bedrohung erkannt"

        # Aktualisiere Nutzungsstatistiken
        self.key_usage[api_key] += 1
        self.key_last_used[api_key] = datetime.now()

        # Erfolgreiche Authentifizierung loggen
        self._log_audit_event("authentication_success", {
            "api_key_hash": self._hash_string(api_key),
            "ip": client_ip,
            "endpoint": request_data.get("endpoint", "unknown")
        })

        return True, "Authentifizierung erfolgreich"

    def _check_rate_limits(self, client_ip: str, api_key: str) -> bool:
        """
        Prüft Rate Limiting für IP und API-Schlüssel
        """
        current_time = time.time()

        # Bereinige alte Einträge (älter als 1 Stunde)
        self.request_history[client_ip] = deque(
            [req for req in self.request_history[client_ip] if current_time - req < 3600]
        )

        # Prüfe Anfragen pro Minute
        recent_requests = [req for req in self.request_history[client_ip] if current_time - req < 60]
        if len(recent_requests) >= self.rate_limits["requests_per_minute"]:
            return False

        # Prüfe Anfragen pro Stunde
        hourly_requests = len(self.request_history[client_ip])
        if hourly_requests >= self.rate_limits["requests_per_hour"]:
            return False

        # Prüfe gleichzeitige Anfragen
        if self.active_requests[client_ip] >= self.rate_limits["concurrent_requests"]:
            return False

        # Anfrage registrieren
        self.request_history[client_ip].append(current_time)
        self.active_requests[client_ip] += 1

        return True

    def _check_content_safety(self, request_data: Dict) -> bool:
        """
        Prüft Inhalt auf Sicherheitsverstöße
        """
        content = str(request_data).lower()

        # Prüfe auf verbotene Keywords
        for filter_name, patterns in self.content_filters.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    logger.warning(f"Content-Filter '{filter_name}' ausgelöst: {pattern}")
                    return False

        return True

    def _detect_threats(self, request_data: Dict) -> bool:
        """
        Erkennt potenzielle Bedrohungen
        """
        content = str(request_data).lower()

        # Prüfe auf bekannte Threat-Patterns
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self.suspicious_activities.append({
                        "timestamp": datetime.now(),
                        "threat_type": threat_type,
                        "pattern": pattern,
                        "content_hash": self._hash_string(content)
                    })
                    return True

        return False

    def _initialize_content_filters(self) -> Dict[str, List[str]]:
        """
        Initialisiert Content-Filter
        """
        return {
            "hate_speech": [
                r"\b(hass|diskriminierung|vorurteil)\b",
                r"\b(rassistisch|sexistisch|homophob)\b"
            ],
            "violence": [
                r"\b(gewalt|mord|töten|verletzen)\b",
                r"\b(waffe|bombe|anschlag)\b"
            ],
            "illegal_activities": [
                r"\b(drogen|illegal|schmuggel)\b",
                r"\b(hacken|virus|malware)\b"
            ],
            "political_extremism": [
                r"\b(nazi|faschist|extremist)\b",
                r"\b(revolution|umsturz|putsch)\b"
            ],
            "personal_data": [
                r"\b(passwort|pin|ssn|social.security)\b",
                r"\b(kreditkarte|bankdaten|iban)\b"
            ]
        }

    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """
        Lädt Threat-Patterns
        """
        return {
            "sql_injection": [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\b.*\b(FROM|INTO|TABLE)\b)",
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bOR\b.*\d+\s*=\s*\d+)"
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*="
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ],
            "command_injection": [
                r"[;&|`$()<>]",
                r"\b(rm|del|format|shutdown)\b"
            ]
        }

    def _load_geoip_database(self):
        """Lädt GeoIP-Datenbank für Standort-basierte Sicherheit"""
        try:
            # Hier würde die GeoIP-Datenbank geladen werden
            # self.geoip_db = geoip2.database.Reader('/path/to/GeoLite2-City.mmdb')
            pass
        except:
            logger.warning("GeoIP-Datenbank nicht verfügbar")

    def get_geolocation_info(self, ip_address: str) -> Optional[Dict]:
        """
        Ermittelt Standort-Informationen für eine IP-Adresse
        """
        if not self.geoip_db:
            return None

        try:
            response = self.geoip_db.city(ip_address)
            return {
                "country": response.country.name,
                "city": response.city.name,
                "latitude": response.location.latitude,
                "longitude": response.location.longitude
            }
        except:
            return None

    def _log_security_event(self, event_type: str, details: Dict):
        """
        Loggt Sicherheitsereignisse
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "security_event",
            "event_type": event_type,
            "details": details
        }

        self.audit_queue.put(event)
        logger.warning(f"🚨 Sicherheitsereignis: {event_type}")

    def _log_audit_event(self, event_type: str, details: Dict):
        """
        Loggt Audit-Ereignisse
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "audit_event",
            "event_type": event_type,
            "details": details
        }

        self.audit_queue.put(event)

    def _audit_logging_loop(self):
        """Verarbeitet Audit-Logs im Hintergrund"""
        while True:
            try:
                event = self.audit_queue.get(timeout=1)
                self.audit_log.append(event)

                # Schreibe wichtige Ereignisse in Datei
                if event["type"] == "security_event":
                    self._write_security_log(event)

                self.audit_queue.task_done()

            except:
                continue

    def _write_security_log(self, event: Dict):
        """Schreibt Sicherheitsereignisse in Log-Datei"""
        log_file = f"security_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
            f.write("\n")

    def _security_monitoring_loop(self):
        """Überwacht Sicherheitsmetriken"""
        while True:
            try:
                # Analysiere verdächtige Aktivitäten
                self._analyze_suspicious_activities()

                # Prüfe auf Brute-Force-Angriffe
                self._check_brute_force_attempts()

                # Aktualisiere Blocklisten
                self._update_blocklists()

                time.sleep(300)  # Alle 5 Minuten

            except Exception as e:
                logger.error(f"Sicherheits-Monitoring Fehler: {e}")
                time.sleep(60)

    def _analyze_suspicious_activities(self):
        """Analysiert verdächtige Aktivitäten"""
        # Zähle Ereignisse pro IP
        ip_counts = defaultdict(int)
        for activity in self.suspicious_activities:
            ip_counts[activity.get("ip", "unknown")] += 1

        # Blockiere IPs mit zu vielen verdächtigen Aktivitäten
        for ip, count in ip_counts.items():
            if count >= 10:  # Schwellenwert
                self.blocked_ips.add(ip)
                logger.warning(f"🚫 IP {ip} blockiert wegen {count} verdächtigen Aktivitäten")

    def _check_brute_force_attempts(self):
        """Prüft auf Brute-Force-Angriffe"""
        current_time = datetime.now()

        # Bereinige alte Einträge
        for ip in list(self.failed_login_attempts.keys()):
            self.failed_login_attempts[ip] = [
                attempt for attempt in self.failed_login_attempts[ip]
                if (current_time - attempt).seconds < 3600  # Letzte Stunde
            ]
            if not self.failed_login_attempts[ip]:
                del self.failed_login_attempts[ip]

        # Blockiere IPs mit zu vielen fehlgeschlagenen Versuchen
        for ip, attempts in self.failed_login_attempts.items():
            if len(attempts) >= self.brute_force_threshold:
                self.blocked_ips.add(ip)
                logger.warning(f"🚫 IP {ip} blockiert wegen Brute-Force-Versuch")

    def _update_blocklists(self):
        """Aktualisiert Blocklisten"""
        # Hier könnte eine Integration mit externen Threat-Intelligence-Feeds erfolgen
        pass

    def _cleanup_loop(self):
        """Bereinigt alte Daten"""
        while True:
            try:
                # Bereinige alte Request-Historie
                current_time = time.time()
                for ip in list(self.request_history.keys()):
                    self.request_history[ip] = deque(
                        [req for req in self.request_history[ip] if current_time - req < 86400]  # 24 Stunden
                    )
                    if not self.request_history[ip]:
                        del self.request_history[ip]

                # Bereinige aktive Requests (Fallback falls nicht ordnungsgemäß dekrementiert)
                for ip in list(self.active_requests.keys()):
                    if self.active_requests[ip] <= 0:
                        del self.active_requests[ip]

                time.sleep(3600)  # Alle Stunden

            except Exception as e:
                logger.error(f"Cleanup Fehler: {e}")
                time.sleep(300)

    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """
        Generiert einen neuen API-Schlüssel

        Args:
            user_id: Benutzer-ID
            permissions: Liste der Berechtigungen

        Returns:
            str: Der generierte API-Schlüssel
        """
        # Generiere sicheren API-Schlüssel
        api_key = secrets.token_urlsafe(32)

        # Speichere Schlüssel-Informationen
        self.api_keys[api_key] = {
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "active": True,
            "permissions": permissions or ["read"],
            "usage_count": 0
        }

        self._save_api_keys()
        logger.info(f"✅ Neuer API-Schlüssel für User {user_id} generiert")

        return api_key

    def revoke_api_key(self, api_key: str):
        """
        Widerruft einen API-Schlüssel
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            self.api_keys[api_key]["revoked_at"] = datetime.now().isoformat()
            self._save_api_keys()
            logger.info(f"🚫 API-Schlüssel widerrufen: {self._hash_string(api_key)}")

    def _load_api_keys(self) -> Dict:
        """Lädt API-Schlüssel aus Datei"""
        keys_file = "api_keys.json"
        if os.path.exists(keys_file):
            with open(keys_file, "r") as f:
                return json.load(f)
        return {}

    def _save_api_keys(self):
        """Speichert API-Schlüssel in Datei"""
        with open("api_keys.json", "w") as f:
            json.dump(self.api_keys, f, indent=2)

    def _hash_string(self, text: str) -> str:
        """Erstellt einen Hash für Logging-Zwecke"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get_security_report(self) -> Dict:
        """
        Erstellt einen Sicherheitsbericht
        """
        return {
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k, v in self.api_keys.items() if v.get("active", True)]),
            "blocked_ips": len(self.blocked_ips),
            "blocked_users": len(self.blocked_users),
            "suspicious_activities_today": len([
                a for a in self.suspicious_activities
                if (datetime.now() - a["timestamp"]).days < 1
            ]),
            "audit_events_today": len([
                e for e in self.audit_log
                if e["timestamp"].startswith(datetime.now().strftime("%Y-%m-%d"))
            ]),
            "rate_limit_violations": sum(1 for events in self.request_history.values()
                                       if len([e for e in events if time.time() - e < 3600]) >
                                       self.rate_limits["requests_per_hour"])
        }

# Globale Instanz des Sicherheitssystems
security_system = AdvancedSecuritySystem()

def get_security_system():
    """Gibt die globale Instanz des Sicherheitssystems zurück"""
    return security_system