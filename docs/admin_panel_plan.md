# Admin Panel Funktionsübersicht
# Bundeskanzler KI - Admin-Panel Erweiterung

## Admin-Panel Funktionen:

### 1. Admin-Authentifizierung
- Separate Admin-Login mit erhöhten Rechten
- Admin-Token mit speziellen Claims
- Rollen-basierte Berechtigung (admin, user)

### 2. System-Status Dashboard
- Server Health: CPU, Memory, Disk Usage
- API Performance: Request Count, Response Times
- Active Connections & Rate Limits
- Memory System Status: Kurz-/Langzeitgedächtnis Auslastung

### 3. Live Log-Viewer
- Real-time Log-Stream von allen Log-Files
- Filter nach Level, Zeitraum, Module
- Download-Funktion für Log-Exports
- Search & Highlight in Logs

### 4. Benutzer-Management
- Liste aller Benutzer mit Status
- Benutzer erstellen/bearbeiten/deaktivieren
- Zugriffsrechte und API-Limits verwalten
- Login-Historie und Activity-Tracking

### 5. Memory-Management Tools
- Memory-Statistiken und Auslastung
- Memory komplett leeren (mit Bestätigung)
- Memory exportieren/importieren
- Duplikate bereinigen
- Memory-Backup erstellen

### 6. System-Konfiguration
- API-Limits: Rate Limits, Max Request Size
- Memory-Einstellungen: Größe, Retention, Duplikat-Schwellwerte
- Logging-Level: Debug, Info, Warning, Error
- Sicherheits-Einstellungen: Token-Expiry, Allowed IPs

### 7. API-Statistiken
- Request-Verteilung nach Endpunkten
- Response-Zeit Trends
- Fehler-Rate und Error-Tracking
- Top Users und Activity-Patterns

## Technische Umsetzung:
- FastAPI Admin-Endpoints: /admin/*
- Streamlit Admin-Interface: Admin-Bereich in Web-GUI
- Admin-Rolle: JWT-Token mit admin-claim
- Database: JSON-Files für User-Management
- Real-time: WebSocket für Live-Updates