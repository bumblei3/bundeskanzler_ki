# 🚀 Bundeskanzler KI - API Dokumentation

## 📋 Übersicht

Die Bundeskanzler KI bietet eine umfassende REST API für politische Fragen und Beratung. Die API ist mit FastAPI implementiert und bietet automatische OpenAPI-Dokumentation.

## 🔧 Technische Details

- **Framework**: FastAPI
- **Authentifizierung**: JWT Token
- **Rate Limiting**: Implementiert
- **Dokumentation**: Automatisch generiert unter `/docs`
- **Gesundheitscheck**: `/health` Endpoint

## 🎯 Hauptkomponenten

### 🤖 KI-System Endpunkte

#### POST `/query`
Haupt-Query-Endpunkt für politische Fragen mit Fact-Checking.

**Request Body:**
```json
{
  "query": "Was ist die aktuelle Klimapolitik Deutschlands?",
  "fact_check": true,
  "language": "de"
}
```

**Response:**
```json
{
  "answer": "Die aktuelle Klimapolitik Deutschlands...",
  "confidence": 0.85,
  "sources": [
    {
      "name": "Bundesregierung",
      "url": "https://bundesregierung.de",
      "confidence": 0.9
    }
  ],
  "processing_time": 0.34,
  "language": "de"
}
```

#### GET `/query/history`
Abrufen des Query-Verlaufs.

**Response:**
```json
{
  "queries": [
    {
      "id": "123",
      "query": "Klimapolitik Frage",
      "timestamp": "2025-09-16T13:30:00Z",
      "confidence": 0.85
    }
  ]
}
```

### 🔐 Authentifizierung

#### POST `/auth/token`
JWT Token für normale Benutzer generieren.

**Request:**
```json
{
  "username": "user",
  "password": "password"
}
```

#### POST `/auth/admin-token`
JWT Token für Administratoren generieren.

### 👨‍💼 Admin-Endpunkte

#### GET `/admin/system-stats`
System-Statistiken abrufen.

**Response:**
```json
{
  "gpu_usage": 15.2,
  "memory_usage": 2.1,
  "active_queries": 3,
  "total_queries": 1250
}
```

#### GET `/admin/logs/{log_type}`
Logs abrufen (api, memory, security).

#### POST `/admin/memory/clear`
Arbeitsspeicher bereinigen.

#### GET `/admin/memory/stats`
Speicher-Statistiken abrufen.

#### POST `/admin/cache/clear`
Cache bereinigen.

#### GET `/admin/cache/stats`
Cache-Statistiken abrufen.

#### GET `/admin/performance/stats`
Performance-Metriken abrufen.

### 👥 Benutzerverwaltung

#### GET `/admin/users`
Alle Benutzer auflisten.

#### POST `/admin/users`
Neuen Benutzer erstellen.

**Request:**
```json
{
  "username": "neuer_user",
  "email": "user@example.com",
  "role": "user"
}
```

#### DELETE `/admin/users/{user_id}`
Benutzer löschen.

### ⚙️ Konfiguration

#### GET `/admin/config`
Systemkonfiguration abrufen.

#### PUT `/admin/config`
Systemkonfiguration aktualisieren.

### 🧠 Gedächtnis-System

#### POST `/memory/add`
Neue Information zum Gedächtnis hinzufügen.

**Request:**
```json
{
  "content": "Neue politische Information",
  "category": "politik",
  "importance": 0.8
}
```

#### POST `/memory/search`
Im Gedächtnis suchen.

**Request:**
```json
{
  "query": "Klimapolitik",
  "limit": 10
}
```

## 🔍 Fact-Checking Integration

Das Fact-Checking System ist in alle Query-Endpunkte integriert:

- **Automatische Validierung**: Jede Antwort wird gegen 6 vertrauenswürdige Quellen validiert
- **Konfidenz-Scoring**: 75%+ Durchschnitts-Konfidenz erreicht
- **Quellen-Transparenz**: Alle Quellen werden in der Antwort aufgeführt
- **Performance-optimiert**: Caching-System für wiederholte Queries

## 🌍 Multilingual Support

- **Unterstützte Sprachen**: Deutsch, Englisch, Italienisch, Spanisch, Französisch
- **Automatische Erkennung**: Sprache wird automatisch erkannt
- **Fallback**: Deutsche Ausgabe bei unbekannten Sprachen

## 📊 Monitoring & Health Checks

#### GET `/admin/health`
Gesundheitsstatus des Systems.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0",
  "uptime": "2d 4h 30m",
  "gpu_status": "active",
  "memory_usage": "2.1GB"
}
```

## 🚀 Verwendung

### Start der API

```bash
# API Server starten
uvicorn core.bundeskanzler_api:app --host 0.0.0.0 --port 8000

# Mit Reload für Entwicklung
uvicorn core.bundeskanzler_api:app --reload --host 0.0.0.0 --port 8000
```

### Automatische Dokumentation

Nach dem Start der API ist die interaktive Dokumentation verfügbar unter:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔒 Sicherheit

- **JWT-Authentifizierung**: Sichere Token-basierte Authentifizierung
- **Rate Limiting**: Schutz vor Missbrauch
- **Input Validation**: Automatische Validierung aller Eingaben
- **Logging**: Umfassende Sicherheitslogs

## 📈 Performance

- **Query-Zeit**: ~0.2-0.6 Sekunden
- **GPU-Auslastung**: 8-20% (RTX 2070)
- **Speicherverbrauch**: ~1.7GB
- **Concurrent Queries**: Bis zu 8 parallele Anfragen

## 🧪 Testing

```bash
# API Tests ausführen
python3 -m pytest tests/test_api.py -v

# Integration Tests
python3 test_fact_check_integration.py
```

---

**API Version**: 2.0
**Letztes Update**: 16. September 2025