# 🔍 Debug-System - Bundeskanzler KI
## Erweiterte Fehlerbehebung und Monitoring

**Version 1.1.0** - Vollständig integriert in Web-GUI mit erweitertem API-Tracking

## 📋 Übersicht

Das Debug-System der Bundeskanzler KI bietet umfassende Möglichkeiten zur Fehlerbehebung, Performance-Monitoring und Systemdiagnose. Es ist vollständig in die Web-GUI integriert und bietet Live-Überwachung aller Systemaktivitäten.

## 🎯 Kernfunktionen

### 📝 Strukturiertes Logging
- **5 Log-Levels**: INFO (ℹ️), SUCCESS (✅), WARNING (⚠️), ERROR (❌), DEBUG (🔍)
- **Automatische Erfassung**: Alle Systemereignisse werden strukturiert protokolliert
- **Zeitstempel**: Präzise Timing-Informationen für alle Ereignisse
- **Kontext-Informationen**: Zusätzliche Daten zu jedem Log-Eintrag

### 🌐 API-Call-Tracking
- **Automatische Erfassung**: Alle API-Aufrufe werden getrackt
- **Performance-Metriken**: Response-Zeiten und Durchsatz-Messungen
- **Fehlerdiagnose**: Detaillierte Fehlerberichte mit Stack-Traces
- **Status-Monitoring**: Live-Überwachung der API-Verfügbarkeit

### 💻 Live Debug-Konsole
- **Web-GUI Integration**: Kollabierbare Debug-Anzeige
- **Echtzeit-Updates**: Live-Anzeige aller Systemaktivitäten
- **Filter-Optionen**: Nach Log-Level und Zeitraum filtern
- **Export-Funktionen**: Debug-Daten können exportiert werden

## 🏗️ Technische Architektur

### DebugSystem Klasse
```python
class DebugSystem:
    def __init__(self):
        self.enabled = True
        self.messages = []  # Liste aller Debug-Nachrichten
        self.api_calls = []  # Liste aller API-Calls
        self.start_time = time.time()

    def log(self, level: DebugLevel, message: str, data=None):
        """Fügt eine Debug-Nachricht hinzu"""

    def log_api_call(self, endpoint: str, method: str, status_code: int,
                    response_time: float, error=None):
        """Erfasst API-Call-Informationen"""
```

### DebugLevel Enum
```python
class DebugLevel(Enum):
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    DEBUG = "🔍"
```

## 📊 Verwendung in der Web-GUI

### Debug-Konsole
Die Debug-Konsole ist in der Admin-Oberfläche verfügbar und zeigt:

1. **Log-Nachrichten**: Alle Systemereignisse in chronologischer Reihenfolge
2. **API-Calls**: Übersicht über alle API-Anfragen mit Status und Timing
3. **Performance-Metriken**: Response-Zeiten und Fehlerquoten
4. **System-Status**: Live-Informationen über CPU, Memory und API-Status

### Filter- und Suchfunktionen
- **Nach Log-Level filtern**: Nur bestimmte Arten von Nachrichten anzeigen
- **Zeitbereich einschränken**: Debug-Daten für bestimmte Zeiträume
- **Textsuche**: Nach bestimmten Begriffen in den Nachrichten suchen
- **API-Endpunkt-Filter**: Nur Calls zu bestimmten Endpunkten anzeigen

## 🔧 API-Integration

### Automatische Erfassung
Das Debug-System erfasst automatisch:

- **Authentifizierungsversuche**: Login-Vorgänge und Token-Generierung
- **API-Anfragen**: Alle REST-API-Calls mit Parametern und Responses
- **Datenbank-Operationen**: Memory-System Zugriffe und Änderungen
- **Modell-Interaktionen**: KI-Modell Aufrufe und Ergebnisse
- **System-Metriken**: CPU, Memory und GPU-Auslastung

### Performance-Monitoring
- **Response-Zeiten**: Messung der API-Antwortgeschwindigkeit
- **Fehlerquoten**: Prozentsatz fehlgeschlagener Anfragen
- **Durchsatz**: Anzahl der Anfragen pro Zeiteinheit
- **Speicherverbrauch**: Monitoring der Systemressourcen

## 🚨 Fehlerbehebung

### Häufige Probleme

#### 1. API-Verbindungsfehler (401 Unauthorized)
```
ERROR:root:❌ API POST /auth/admin-token -> 401 (0.01s)
```
**Lösung**: Überprüfen Sie die Admin-Credentials (`admin` / `admin123!`)

#### 2. matplotlib Warnungen
```
UserWarning: Attempting to set identical low and high xlims
```
**Lösung**: Automatisch behoben durch robuste Chart-Skalierung

#### 3. Memory-API Fehler (500 Internal Server Error)
```
ERROR:root:❌ API GET /admin/memory/stats -> 500
```
**Lösung**: Überprüfen Sie die API-Authentifizierung und Token-Gültigkeit

### Debug-Modi

#### Entwicklungsmodus
```bash
# Vollständiges Debug-Logging aktivieren
export DEBUG_MODE=true
streamlit run webgui_ki.py
```

#### Produktionsmodus
```bash
# Nur Fehler und Warnungen loggen
export DEBUG_LEVEL=WARNING
streamlit run webgui_ki.py
```

## 📈 Monitoring & Analyse

### Live-Metriken
- **API-Health**: Verfügbarkeit aller Endpunkte
- **Response-Zeiten**: Durchschnittliche und maximale Antwortzeiten
- **Fehlerraten**: Prozentsatz fehlgeschlagener Anfragen
- **Systemauslastung**: CPU, Memory und GPU-Monitoring

### Historische Daten
- **Log-Archivierung**: Automatische Speicherung alter Debug-Daten
- **Performance-Trends**: Langfristige Analyse der Systemleistung
- **Fehler-Patterns**: Erkennung wiederkehrender Probleme

## 🔒 Sicherheit & Datenschutz

### Datenminimierung
- **Automatische Bereinigung**: Alte Debug-Daten werden regelmäßig entfernt
- **Sensible Daten**: Passwörter und Tokens werden nicht in Logs gespeichert
- **IP-Adressen**: Werden anonymisiert in Debug-Ausgaben

### Zugriffskontrolle
- **Admin-only**: Debug-Informationen sind nur für Administratoren zugänglich
- **Verschlüsselte Speicherung**: Debug-Daten werden sicher gespeichert
- **Audit-Logs**: Alle Zugriffe auf Debug-Daten werden protokolliert

## 🚀 Erweiterte Features (zukünftig)

### Geplante Erweiterungen
- **Alert-System**: Automatische Benachrichtigungen bei kritischen Fehlern
- **Performance-Analyse**: Detaillierte Bottleneck-Analyse
- **Remote-Debugging**: Debug-Unterstützung für entfernte Systeme
- **Custom Dashboards**: Individuelle Monitoring-Dashboards

---

## 📞 Support

Bei Problemen mit dem Debug-System:
1. Überprüfen Sie die Debug-Konsole in der Web-GUI
2. Konsultieren Sie die API-Logs: `http://localhost:8001/docs`
3. Verwenden Sie die integrierten Diagnose-Tools
4. Kontaktieren Sie den Systemadministrator

**Letzte Aktualisierung**: 15. September 2025</content>
<parameter name="filePath">/home/tobber/bkki_venv/DEBUG_SYSTEM.md