# Plugin-System für Bundeskanzler KI

## Übersicht

Das Plugin-System der Bundeskanzler KI ist ein modulares Framework, das es ermöglicht, die Funktionalität der KI durch externe Plugins zu erweitern. Es bietet eine saubere API für Plugin-Entwickler und integriert sich nahtlos in die bestehende Architektur.

## Hauptmerkmale

- **Modulare Architektur**: Einfache Erweiterung ohne Änderung des Kerncodes
- **Automatische Plugin-Entdeckung**: Plugins werden automatisch gefunden und geladen
- **Sicherheit**: Integrierte Sicherheitsprüfungen und Plugin-Isolation
- **Hook-System**: Event-basierte Integration mit dem KI-System
- **REST-API**: Vollständige API für Plugin-Management
- **Monitoring**: Umfassende Überwachung von Plugin-Performance und -Sicherheit

## Architektur

### Kernkomponenten

1. **PluginManager**: Zentraler Manager für Plugin-Lebenszyklus
2. **BasePlugin**: Basis-Klasse für alle Plugins
3. **Plugin-API**: REST-Endpunkte für Plugin-Management
4. **Plugin-Integration**: Nahtlose Integration mit bestehenden Systemen

### Plugin-Typen

- **TextProcessingPlugin**: Für Textverarbeitung und -verbesserung
- **ImageProcessingPlugin**: Für Bildverarbeitung und -analyse
- **AudioProcessingPlugin**: Für Audioverarbeitung und Transkription
- **HookPlugin**: Für System-Hooks und Lebenszyklus-Events

## Installation und Setup

### Voraussetzungen

- Python 3.8+
- FastAPI (für API-Integration)
- psutil (für Monitoring-Plugin)

### Automatische Integration

Das Plugin-System ist bereits in die Bundeskanzler-API integriert und wird automatisch beim Start der Anwendung initialisiert.

## Plugin-Entwicklung

### Grundstruktur eines Plugins

```python
from core.plugin_system import BasePlugin, PluginMetadata

class MyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="Mein benutzerdefiniertes Plugin",
            author="Ihr Name",
            license="MIT"
        )

    def initialize(self) -> None:
        """Wird beim Laden des Plugins aufgerufen"""
        self.logger.info("Plugin initialisiert")

    def shutdown(self) -> None:
        """Wird beim Entladen des Plugins aufgerufen"""
        self.logger.info("Plugin beendet")
```

### Plugin-Verzeichnis-Struktur

```
plugins/
├── my_plugin/
│   ├── __init__.py          # Plugin-Implementierung
│   └── config.json          # Optionale Konfiguration
└── another_plugin/
    └── __init__.py
```

### Beispiel-Plugins

Das System enthält bereits drei Beispiel-Plugins:

1. **Text-Verbesserungs-Plugin** (`text_improvement`)
   - Verbessert Grammatik und Stil von Textausgaben
   - Korrigiert häufige Fehler
   - Formelle Kommunikationsstandards

2. **Monitoring-Plugin** (`monitoring`)
   - Überwacht System-Performance
   - Sammelt Metriken über CPU, Speicher, Plugin-Ausführung
   - Generiert Performance-Berichte

3. **Sicherheits-Plugin** (`security`)
   - Überprüft Anfragen auf Sicherheitsrisiken
   - Rate-Limiting und IP-Filterung
   - SQL-Injection und XSS-Schutz

## API-Endpunkte

### Plugin-Management

- `GET /api/plugins/` - Alle Plugins auflisten
- `GET /api/plugins/{name}` - Plugin-Details abrufen
- `POST /api/plugins/{name}/load` - Plugin laden
- `POST /api/plugins/{name}/unload` - Plugin entladen
- `GET /api/plugins/{name}/config` - Plugin-Konfiguration abrufen
- `PUT /api/plugins/{name}/config` - Plugin-Konfiguration aktualisieren
- `POST /api/plugins/{name}/enable` - Plugin aktivieren
- `POST /api/plugins/{name}/disable` - Plugin deaktivieren
- `POST /api/plugins/{name}/execute/{method}` - Plugin-Methode ausführen
- `POST /api/plugins/reload` - Alle Plugins neu laden

### Spezielle Endpunkte

- `GET /api/plugins/types` - Verfügbare Plugin-Typen
- `GET /api/plugins/monitoring/metrics` - Monitoring-Metriken
- `GET /api/plugins/security/report` - Sicherheitsbericht

## Sicherheit

### Plugin-Isolation

- Jedes Plugin läuft in seiner eigenen Sandbox
- Sicherheitsprüfungen vor Plugin-Ausführung
- Ressourcen-Limits und Timeouts

### Zugriffssteuerung

- IP-Whitelist/Blacklist
- Rate-Limiting pro IP
- Anfragegrößen-Beschränkungen

### Bedrohungs-Erkennung

- SQL-Injection-Erkennung
- XSS-Filterung
- Command-Injection-Schutz
- Verdächtige Muster-Erkennung

## Monitoring und Logging

### Metriken

- Plugin-Ausführungszeiten
- System-Ressourcen-Verbrauch
- Sicherheitsereignisse
- Fehler-Raten

### Logging

- Strukturiertes Logging für alle Plugin-Aktivitäten
- Sicherheitsereignis-Protokollierung
- Performance-Monitoring

## Tests

Das Plugin-System enthält umfassende Tests:

```bash
# Alle Tests ausführen
python test_plugin_system.py

# Einzelne Test-Komponenten
python -c "from core.plugin_system import *; print('Plugin-System OK')"
python -c "from core.plugin_api_fastapi import *; print('Plugin-API OK')"
python -c "from core.plugin_integration import *; print('Plugin-Integration OK')"
```

## Erweiterte Konfiguration

### Plugin-Konfiguration

Plugins können über die API oder Konfigurationsdateien konfiguriert werden:

```json
{
  "enabled": true,
  "priority": 100,
  "settings": {
    "custom_option": "value"
  },
  "auto_start": true
}
```

### System-Konfiguration

Das Plugin-System kann über Umgebungsvariablen konfiguriert werden:

```bash
export PLUGIN_DIRS="/opt/plugins,/usr/local/plugins"
export MAX_PLUGINS=50
export PLUGIN_TIMEOUT=30
```

## Fehlerbehebung

### Häufige Probleme

1. **Plugin wird nicht gefunden**
   - Überprüfen Sie die Verzeichnisstruktur
   - Stellen Sie sicher, dass `__init__.py` vorhanden ist
   - Prüfen Sie die Python-Pfad-Konfiguration

2. **Plugin lädt nicht**
   - Überprüfen Sie die Logs auf Import-Fehler
   - Prüfen Sie Plugin-Abhängigkeiten
   - Validieren Sie die Plugin-Metadaten

3. **Sicherheitsfehler**
   - Überprüfen Sie die IP-Whitelist
   - Prüfen Sie Rate-Limiting-Einstellungen
   - Validieren Sie Anfrage-Inhalte

### Debug-Modus

Aktivieren Sie detailliertes Logging:

```python
import logging
logging.getLogger('plugin').setLevel(logging.DEBUG)
```

## Beitrag und Entwicklung

### Plugin-Entwicklungsrichtlinien

1. Erben Sie von der entsprechenden Basis-Klasse
2. Implementieren Sie alle abstrakten Methoden
3. Verwenden Sie das Logging-System des Plugins
4. Behandeln Sie Fehler angemessen
5. Dokumentieren Sie Ihre Plugin-API

### Code-Qualität

- Verwenden Sie Type Hints
- Schreiben Sie umfassende Docstrings
- Implementieren Sie Unit-Tests
- Folgen Sie PEP 8 Stilrichtlinien

## Lizenz

Das Plugin-System ist unter der MIT-Lizenz veröffentlicht.

## Support

Bei Fragen oder Problemen:

1. Überprüfen Sie die Logs auf Fehlermeldungen
2. Testen Sie mit dem bereitgestellten Test-Skript
3. Konsultieren Sie die API-Dokumentation
4. Erstellen Sie ein Issue im Projekt-Repository

---

**Hinweis**: Dieses Plugin-System ist speziell für die Bundeskanzler KI entwickelt und optimiert. Für allgemeine Python-Anwendungen sollten Sie andere Plugin-Frameworks in Betracht ziehen.