# 🤖 Bundeskanzler KI - Anleitung zum Ausprobieren

## 🚀 Schnellstart

### 1️⃣ API starten
```bash
cd /home/tobber/bkki_venv
source bin/activate
python3 bundeskanzler_api.py
```

### 2️⃣ Admin-Panel starten (neues Terminal)
```bash
cd /home/tobber/bkki_venv
source bin/activate
streamlit run webgui_ki.py --server.port 8501 --server.address 0.0.0.0
```

### 3️⃣ Admin-Panel öffnen
- **URL:** http://localhost:8501
- **Login:** admin / admin123!

## 💬 Verschiedene Wege zum Ausprobieren

### 🎯 Methode 1: Admin-Panel (Empfohlen!)
```bash
streamlit run webgui_ki.py
```
**Features:**
- � **Dashboard**: Live-System-Metriken
- 👥 **Benutzer-Management**: User verwalten
- � **Log-Viewer**: Live-Logs mit Filterung
- 💾 **Memory-Management**: Statistiken und Verwaltung (neu behoben!)
- ⚙️ **Konfiguration**: System-Einstellungen

### 🌐 Methode 2: Web-Browser (Swagger UI)
1. API starten (siehe oben)
2. Browser öffnen: http://localhost:8000/docs
3. **Features:**
   - 📖 Automatische API-Dokumentation
   - 🧪 Direkte API-Tests im Browser
   - 🔐 Authentifizierung testen

### 🔧 Methode 3: curl Kommandos
```bash
# Health Check
curl http://localhost:8000/health

# Admin-Login
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/admin-token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123!" | jq -r .access_token)

# Memory-Stats (neu behoben!)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/memory/stats

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Wie steht es um die Klimapolitik?"}'
```

### 🧪 Methode 4: Vollständiger Test
```bash
python direct_test.py
```

## 💡 Was du ausprobieren kannst

### 🗣️ Chat-Themen
- **Klimapolitik**: "Wie steht es um den Klimaschutz in Deutschland?"
- **Wirtschaft**: "Wie entwickelt sich die deutsche Wirtschaft?"
- **Digitalisierung**: "Welche Digitalisierungsmaßnahmen plant die Regierung?"
- **Energiewende**: "Können Sie mir zur Energiewende etwas sagen?"
- **Soziales**: "Was unternimmt die Regierung für soziale Gerechtigkeit?"

### 🧠 Memory-System
```bash
# In der Demo:
/memory Deutschland plant bis 2030 eine führende Rolle in der KI-Entwicklung
/search KI Deutschland
/stats
```

### 🔗 Webhooks testen
```bash
curl -X POST "http://localhost:8000/webhook/news_update" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Neue KI-Strategie angekündigt",
    "source": "Bundespressekonferenz",
    "timestamp": "2025-09-14T12:00:00"
  }'
```

## 🎯 Demo-Beispiele

### Beispiel-Konversation:
```
Sie: Hallo! Wie geht es Deutschland?
🤖: Vielen Dank für Ihre Anfrage. Deutschland steht vor wichtigen 
    Herausforderungen und Chancen...

Sie: /memory Deutschland investiert 10 Milliarden in KI-Forschung
✅ Erinnerung hinzugefügt!

Sie: Was plant Deutschland für Künstliche Intelligenz?
🤖: Deutschland hat eine umfassende KI-Strategie entwickelt...

Sie: /search KI Forschung
🔍 Gefunden 1 Ergebnisse:
   1. Deutschland investiert 10 Milliarden in KI-Forschung...
```

## 🔐 Anmeldedaten

### Admin-Panel (Streamlit)
- **Username**: `admin`
- **Password**: `admin123!`

### API-Zugang (für Entwickler)
- **Username**: `bundeskanzler`
- **Password**: `ki2025`

## 📊 URLs

### Admin-Panel
- **Streamlit GUI**: http://localhost:8501 (Empfohlen!)
- **HTML Admin-Panel**: http://localhost:8000/admin (Einfach)

### API-Endpunkte
- **API Base**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health

### Admin-API (erfordert Token)
- **Login**: `POST /auth/admin-token`
- **Memory-Stats**: `GET /admin/memory/stats` ✅ (Neu behoben!)
- **System-Stats**: `GET /admin/system-stats`
- **Logs**: `GET /admin/logs/{type}`
- **Users**: `GET /admin/users`

## 🚨 Problemlösung

### API startet nicht?
```bash
# Prüfe ob Port belegt ist
lsof -i :8000

# Stoppe andere uvicorn Prozesse
pkill -f uvicorn

# Starte neu
python -m uvicorn bundeskanzler_api:app --host 0.0.0.0 --port 8000 --reload
```

### Demo verbindet nicht?
- ✅ Stelle sicher, dass die API läuft (siehe oben)
- ✅ Warte 5-10 Sekunden nach API-Start
- ✅ Prüfe dass kein Firewall die Verbindung blockiert

### Authentifizierung schlägt fehl?
- ✅ Verwende exakt: `bundeskanzler` / `ki2025`
- ✅ Prüfe dass die API vollständig gestartet ist

## 🎉 Viel Spaß beim Ausprobieren!

Die Bundeskanzler KI ist bereit für deine Fragen! 🤖