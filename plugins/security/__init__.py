#!/usr/bin/env python3
"""
Sicherheits-Plugin für Bundeskanzler KI
Überwacht und schützt das System vor Sicherheitsbedrohungen
"""

import hashlib
import re
import time
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
from core.plugin_system import HookPlugin, PluginMetadata, PluginSecurityError

class SecurityPlugin(HookPlugin):
    """
    Plugin zur Sicherheit und Überwachung

    Überwacht Anfragen auf Sicherheitsrisiken, verhindert Angriffe und protokolliert
    verdächtige Aktivitäten.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="security",
            version="1.0.0",
            description="Überwacht und schützt das System vor Sicherheitsbedrohungen",
            author="Bundeskanzler KI Team",
            license="MIT",
            tags=["security", "monitoring", "protection", "threat-detection"],
            dependencies=[]
        )

    def initialize(self) -> None:
        """Initialisiert das Plugin"""
        self.logger.info("Sicherheits-Plugin initialisiert")

        # Sicherheits-Konfiguration
        self.max_requests_per_minute = self._config.settings.get('max_requests_per_minute', 60)
        self.max_request_size = self._config.settings.get('max_request_size', 1000000)  # 1MB
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = {}

        # Sicherheits-Statistiken
        self.security_events = []
        self.blocked_requests = 0
        self.suspicious_requests = 0

        # Whitelist und Blacklist
        self.ip_whitelist: Set[str] = set(self._config.settings.get('ip_whitelist', []))
        self.ip_blacklist: Set[str] = set(self._config.settings.get('ip_blacklist', []))

    def shutdown(self) -> None:
        """Beendet das Plugin"""
        self.logger.info("Sicherheits-Plugin beendet")

    def check_request_security(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Überprüft eine Anfrage auf Sicherheitsrisiken

        Args:
            request_data: Daten der eingehenden Anfrage

        Returns:
            Dictionary mit Sicherheitsbewertung

        Raises:
            PluginSecurityError: Bei kritischen Sicherheitsverletzungen
        """
        security_result = {
            'safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': []
        }

        # IP-basierte Überprüfungen
        client_ip = request_data.get('client_ip', 'unknown')
        if not self._check_ip_security(client_ip):
            security_result['safe'] = False
            security_result['risk_level'] = 'high'
            security_result['issues'].append('blocked_ip')
            raise PluginSecurityError(f"IP {client_ip} ist blockiert")

        # Rate-Limiting prüfen
        if not self._check_rate_limit(client_ip):
            security_result['safe'] = False
            security_result['risk_level'] = 'medium'
            security_result['issues'].append('rate_limit_exceeded')
            security_result['recommendations'].append('Warten Sie einen Moment vor der nächsten Anfrage')

        # Inhaltsbasierte Überprüfungen
        content_issues = self._check_content_security(request_data)
        if content_issues:
            security_result['issues'].extend(content_issues)
            if any(issue in ['sql_injection', 'xss', 'command_injection'] for issue in content_issues):
                security_result['safe'] = False
                security_result['risk_level'] = 'high'
            else:
                security_result['risk_level'] = 'medium'

        # Größenbeschränkungen prüfen
        if not self._check_request_size(request_data):
            security_result['safe'] = False
            security_result['risk_level'] = 'medium'
            security_result['issues'].append('request_too_large')

        # Sicherheitsereignis protokollieren
        if not security_result['safe'] or security_result['risk_level'] != 'low':
            self._log_security_event(request_data, security_result)

        return security_result

    def block_ip(self, ip_address: str, reason: str = "manual_block") -> None:
        """
        Blockiert eine IP-Adresse

        Args:
            ip_address: Die zu blockierende IP-Adresse
            reason: Grund für die Blockierung
        """
        self.blocked_ips.add(ip_address)
        self.logger.warning(f"IP {ip_address} blockiert: {reason}")

        # Ereignis protokollieren
        self._log_security_event({
            'client_ip': ip_address,
            'type': 'ip_block'
        }, {
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

    def unblock_ip(self, ip_address: str) -> None:
        """
        Entblockiert eine IP-Adresse

        Args:
            ip_address: Die zu entblockierende IP-Adresse
        """
        self.blocked_ips.discard(ip_address)
        self.logger.info(f"IP {ip_address} entblockiert")

    def get_security_report(self) -> Dict[str, Any]:
        """
        Erstellt einen Sicherheitsbericht

        Returns:
            Dictionary mit Sicherheitsstatistiken
        """
        recent_events = self._get_recent_security_events(hours=24)

        return {
            'total_blocked_requests': self.blocked_requests,
            'total_suspicious_requests': self.suspicious_requests,
            'blocked_ips_count': len(self.blocked_ips),
            'recent_events': recent_events,
            'risk_assessment': self._assess_overall_risk(),
            'recommendations': self._generate_security_recommendations()
        }

    def on_request_start(self, request_data: Dict[str, Any]) -> None:
        """Hook für Anfrage-Start - Sicherheitsprüfung"""
        try:
            security_check = self.check_request_security(request_data)
            request_data['security_check'] = security_check

            if not security_check['safe']:
                self.blocked_requests += 1
                self.logger.warning(f"Unsichere Anfrage blockiert: {security_check['issues']}")

        except PluginSecurityError as e:
            self.logger.error(f"Sicherheitsverletzung: {e}")
            request_data['security_blocked'] = True
            raise

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Hook für Fehler - Sicherheitsanalyse"""
        if isinstance(error, PluginSecurityError):
            self.logger.critical(f"Sicherheitsfehler: {error}")
        else:
            # Überprüfe, ob der Fehler sicherheitsrelevant sein könnte
            error_message = str(error).lower()
            if any(keyword in error_message for keyword in ['injection', 'attack', 'unauthorized', 'forbidden']):
                self.logger.warning(f"Potenziell sicherheitsrelevanter Fehler: {error}")
                self._log_security_event(context, {
                    'error_type': 'suspicious_error',
                    'error_message': str(error)
                })

    def _check_ip_security(self, ip_address: str) -> bool:
        """Prüft die Sicherheit einer IP-Adresse"""
        if ip_address in self.ip_blacklist:
            return False

        if self.ip_whitelist and ip_address not in self.ip_whitelist:
            return False

        if ip_address in self.blocked_ips:
            return False

        return True

    def _check_rate_limit(self, ip_address: str) -> bool:
        """Prüft Rate-Limiting für eine IP-Adresse"""
        current_time = time.time()

        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []

        # Entferne alte Einträge (älter als 1 Minute)
        self.rate_limits[ip_address] = [
            timestamp for timestamp in self.rate_limits[ip_address]
            if current_time - timestamp < 60
        ]

        # Prüfe, ob Limit überschritten
        if len(self.rate_limits[ip_address]) >= self.max_requests_per_minute:
            return False

        # Neue Anfrage hinzufügen
        self.rate_limits[ip_address].append(current_time)
        return True

    def _check_content_security(self, request_data: Dict[str, Any]) -> List[str]:
        """Prüft den Inhalt einer Anfrage auf Sicherheitsrisiken"""
        issues = []
        content = str(request_data.get('content', ''))

        # SQL-Injection prüfen
        if self._contains_sql_injection(content):
            issues.append('sql_injection')

        # XSS prüfen
        if self._contains_xss(content):
            issues.append('xss')

        # Command-Injection prüfen
        if self._contains_command_injection(content):
            issues.append('command_injection')

        # Verdächtige Muster prüfen
        for pattern_name, pattern in self.suspicious_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f'suspicious_pattern_{pattern_name}')

        if issues:
            self.suspicious_requests += 1

        return issues

    def _check_request_size(self, request_data: Dict[str, Any]) -> bool:
        """Prüft die Größe einer Anfrage"""
        content = str(request_data.get('content', ''))
        return len(content.encode('utf-8')) <= self.max_request_size

    def _contains_sql_injection(self, content: str) -> bool:
        """Prüft auf SQL-Injection-Muster"""
        sql_patterns = [
            r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b.*;',
            r'\bUNION\s+SELECT\b',
            r';\s*--',
            r';\s*/\*.*\*/',
            r'\bOR\s+1\s*=\s*1\b',
            r'\bAND\s+1\s*=\s*1\b'
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in sql_patterns)

    def _contains_xss(self, content: str) -> bool:
        """Prüft auf XSS-Muster"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>'
        ]

        return any(re.search(pattern, content, re.IGNORECASE | re.DOTALL) for pattern in xss_patterns)

    def _contains_command_injection(self, content: str) -> bool:
        """Prüft auf Command-Injection-Muster"""
        command_patterns = [
            r'[;&|`$]\s*(ls|cat|rm|mkdir|cd|pwd|whoami|id)\b',
            r'\b(echo|eval|exec|system|popen)\s*\(',
            r'\\x[0-9a-fA-F]{2}',
            r'\$\{.*\}'
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in command_patterns)

    def _load_suspicious_patterns(self) -> Dict[str, str]:
        """Lädt verdächtige Muster"""
        return {
            'suspicious_words': r'\b(hack|exploit|attack|malware|virus|trojan)\b',
            'encoded_content': r'%[0-9a-fA-F]{2}',
            'path_traversal': r'\.\./|\.\.\\',
            'admin_access': r'\b(admin|root|superuser)\b',
            'sensitive_data': r'\b(password|secret|key|token|credential)\b'
        }

    def _log_security_event(self, request_data: Dict[str, Any], security_result: Dict[str, Any]) -> None:
        """Protokolliert ein Sicherheitsereignis"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'client_ip': request_data.get('client_ip', 'unknown'),
            'request_type': request_data.get('type', 'unknown'),
            'security_result': security_result
        }

        self.security_events.append(event)

        # Behalte nur die letzten 1000 Ereignisse
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

    def _get_recent_security_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Gibt kürzliche Sicherheitsereignisse zurück"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = []

        for event in self.security_events:
            if datetime.fromisoformat(event['timestamp']) > cutoff_time:
                recent_events.append(event)

        return recent_events

    def _assess_overall_risk(self) -> str:
        """Bewertet das Gesamtrisiko"""
        recent_events = self._get_recent_security_events(hours=1)

        high_risk_events = sum(1 for event in recent_events
                              if event['security_result'].get('risk_level') == 'high')
        medium_risk_events = sum(1 for event in recent_events
                                if event['security_result'].get('risk_level') == 'medium')

        if high_risk_events > 5 or self.blocked_requests > 10:
            return "high"
        elif high_risk_events > 0 or medium_risk_events > 10:
            return "medium"
        else:
            return "low"

    def _generate_security_recommendations(self) -> List[str]:
        """Generiert Sicherheitsempfehlungen"""
        recommendations = []
        risk_level = self._assess_overall_risk()

        if risk_level == "high":
            recommendations.extend([
                "Sofortige Überprüfung der Sicherheitsereignisse erforderlich",
                "Mögliche IP-Blockierungen überprüfen",
                "Firewall-Regeln überprüfen"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Regelmäßige Überwachung der Sicherheitsereignisse empfohlen",
                "Rate-Limiting Einstellungen überprüfen",
                "Verdächtige Muster analysieren"
            ])

        if self.blocked_requests > 0:
            recommendations.append(f"{self.blocked_requests} Anfragen wurden blockiert")

        return recommendations