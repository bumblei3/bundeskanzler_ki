# Web-GUI Test-Infrastruktur

## Übersicht

Diese Test-Infrastruktur bietet umfassende automatisierte Tests für die Bundeskanzler-KI Weboberfläche. Sie umfasst:

- **Automatisierte Web-GUI Tests** (`automated_web_gui_tests.py`)
- **CI/CD Test Runner** (`ci_test_runner.py`)
- **Interaktives Test-Dashboard** (`test_dashboard.py`)
- **GitHub Actions Workflow** (`.github/workflows/web-gui-tests.yml`)

## Schnellstart

### 1. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 2. Tests ausführen
```bash
# Alle Web-GUI Tests
make test-web-gui

# CI/CD Pipeline Tests
make test-web-gui-ci

# Vollständige Test-Suite
make test-all
```

### 3. Dashboard starten
```bash
make dashboard
```

## Test-Komponenten

### 🔧 Automatisierte Web-GUI Tests

**Datei:** `automated_web_gui_tests.py`

**Test-Suites:**
- **Connectivity Tests**: Verbindungsprüfung zu API und GUI
- **Performance Tests**: Antwortzeiten und Durchsatz
- **Functionality Tests**: Kernfunktionalitäten der Weboberfläche
- **UI/UX Tests**: Benutzeroberfläche und Benutzererfahrung
- **Integration Tests**: End-to-End Szenarien

**Verwendung:**
```python
from automated_web_gui_tests import WebGUITestFramework

framework = WebGUITestFramework()
results = framework.run_all_tests()
```

### 🚀 CI/CD Test Runner

**Datei:** `ci_test_runner.py`

**Features:**
- Automatische Service-Verwaltung (API + GUI)
- Umgebungs-Setup und -Cleanup
- Detaillierte Berichterstattung
- Fehlerbehandlung und Timeouts

**Verwendung:**
```bash
python ci_test_runner.py
```

### 📊 Interaktives Dashboard

**Datei:** `test_dashboard.py`

**Features:**
- Live Test-Ergebnisse
- Trend-Analysen
- Suite-Vergleiche
- Detaillierte Statistiken
- Test-Ausführung aus der GUI

**Starten:**
```bash
streamlit run test_dashboard.py
```

## GitHub Actions CI/CD

### Automatische Ausführung

Die Tests werden automatisch ausgeführt bei:
- **Push** auf `main` oder `develop`
- **Pull Requests** zu `main` oder `develop`
- **Täglich** um 2:00 Uhr (Scheduled)
- **Manuell** über GitHub Actions

### Workflow-Konfiguration

**Datei:** `.github/workflows/web-gui-tests.yml`

**Test-Typen:**
- `all`: Alle Tests (Standard)
- `gui`: Nur GUI-Tests
- `api`: Nur API-Tests
- `performance`: Nur Performance-Tests

## Makefile Targets

```bash
# Web-GUI Tests
make test-web-gui          # Alle Web-GUI Tests
make test-web-gui-ci       # CI/CD Pipeline Tests
make test-dashboard        # Dashboard starten

# Vollständige Test-Suite
make test-all             # Alle verfügbaren Tests
make test-dev             # Schnelltests für Entwicklung

# CI/CD
make ci-test              # CI/CD Pipeline ausführen
make ci-deploy            # Deployment (zukünftig)
```

## Test-Ergebnisse

### Speicherort
Test-Ergebnisse werden in `test_results/` gespeichert:
```
test_results/
├── web_gui_test_results_20241201_143022.json
├── web_gui_test_results_20241201_143500.json
└── ...
```

### Format
```json
{
  "connectivity_tests": {
    "total_tests": 4,
    "passed": 4,
    "failed": 0,
    "errors": 0,
    "duration": 2.34,
    "details": [...]
  },
  "performance_tests": {
    "total_tests": 6,
    "passed": 5,
    "failed": 1,
    "errors": 0,
    "duration": 15.67,
    "details": [...]
  }
}
```

## Konfiguration

### Umgebungsvariablen

```bash
# API Konfiguration
API_HOST=localhost
API_PORT=8000

# GUI Konfiguration
GUI_HOST=localhost
GUI_PORT=8502

# Test Konfiguration
TEST_TIMEOUT=30
TEST_RETRIES=3
```

### Service-Ports

- **API**: `http://localhost:8000`
- **GUI**: `http://localhost:8502`
- **Dashboard**: `http://localhost:8503`

## Fehlerbehebung

### Häufige Probleme

1. **Services starten nicht**
   ```bash
   # Manuell starten
   python simple_api.py &
   streamlit run modern_gui.py --server.port 8502 &
   ```

2. **Timeout-Fehler**
   - Erhöhen Sie `TEST_TIMEOUT` in der Konfiguration
   - Überprüfen Sie Service-Health-Endpoints

3. **Abhängigkeiten fehlen**
   ```bash
   pip install -r requirements.txt
   ```

### Debug-Modus

```bash
# Ausführliche Logs
python automated_web_gui_tests.py --verbose

# Einzelne Test-Suite
python -c "
from automated_web_gui_tests import WebGUITestFramework
framework = WebGUITestFramework()
result = framework.test_connectivity()
print(result)
"
```

## Erweiterungen

### Neue Test-Suites hinzufügen

1. **Methode zur WebGUITestFramework-Klasse hinzufügen:**
   ```python
   def test_custom_feature(self) -> TestSuiteResult:
       # Implementierung
       pass
   ```

2. **In `run_all_tests()` einbinden:**
   ```python
   results['custom_tests'] = self.test_custom_feature()
   ```

### Selenium-Integration (zukünftig)

Für browserbasierte Tests kann Selenium hinzugefügt werden:

```bash
pip install selenium
```

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def test_browser_interaction(self):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    # Test-Implementierung
```

## Monitoring und Berichterstattung

### Metriken

- **Test-Abdeckung**: Anzahl ausgeführter Tests
- **Erfolgsrate**: Prozentsatz bestandenener Tests
- **Durchschnittliche Dauer**: Test-Ausführungszeit
- **Trend-Analyse**: Verbesserung/Verschlechterung über Zeit

### Benachrichtigungen

Bei Test-Fehlern können Benachrichtigungen eingerichtet werden:
- Slack-Webhooks
- Email-Benachrichtigungen
- GitHub Issues erstellen

## Beitragende

### Code-Qualität

- Verwenden Sie Type Hints
- Schreiben Sie Docstrings
- Fügen Sie Unit-Tests für neue Funktionen hinzu
- Halten Sie PEP 8 Standards ein

### Test-Richtlinien

- Tests sollten unabhängig voneinander laufen
- Verwenden Sie aussagekräftige Test-Namen
- Testen Sie sowohl positive als auch negative Fälle
- Dokumentieren Sie erwartete Ergebnisse

## Lizenz

Dieses Projekt ist Teil der Bundeskanzler-KI und folgt der gleichen Lizenz.