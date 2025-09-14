#!/bin/bash

# Bundeskanzler KI API Start-Script
# Startet die API mit Produktions-Konfiguration

echo "🚀 Bundeskanzler KI API wird gestartet..."
echo "=" * 50

# Aktiviere Virtual Environment falls vorhanden
if [ -f "bin/activate" ]; then
    echo "📦 Aktiviere Virtual Environment..."
    source bin/activate
fi

# Überprüfe Python-Dependencies
echo "🔍 Überprüfe Dependencies..."
python -c "import fastapi, uvicorn, jose; print('✅ Alle Dependencies verfügbar')" || {
    echo "❌ Dependencies fehlen. Installiere sie mit:"
    echo "   pip install -r requirements-clean.txt"
    exit 1
}

# Überprüfe Bundeskanzler KI Core
echo "🧠 Überprüfe Bundeskanzler KI Core..."
python -c "from bundeskanzler_ki import EnhancedContextProcessor; print('✅ Bundeskanzler KI Core verfügbar')" || {
    echo "❌ Bundeskanzler KI Core nicht gefunden!"
    exit 1
}

# Erstelle Logs-Verzeichnis
mkdir -p logs

# Setze Umgebungsvariablen
export BKKI_LOG_LEVEL=${BKKI_LOG_LEVEL:-"INFO"}
export BKKI_API_HOST=${BKKI_API_HOST:-"0.0.0.0"}
export BKKI_API_PORT=${BKKI_API_PORT:-8000}
export BKKI_API_WORKERS=${BKKI_API_WORKERS:-1}
export BKKI_JWT_SECRET=${BKKI_JWT_SECRET:-"your-secret-key-change-in-production"}

echo "🌐 API-Konfiguration:"
echo "   Host: $BKKI_API_HOST"
echo "   Port: $BKKI_API_PORT"
echo "   Workers: $BKKI_API_WORKERS"
echo "   Log Level: $BKKI_LOG_LEVEL"

# Zeige URLs
echo ""
echo "📋 Verfügbare URLs:"
echo "   🏠 API Status: http://$BKKI_API_HOST:$BKKI_API_PORT/"
echo "   📖 Swagger UI: http://$BKKI_API_HOST:$BKKI_API_PORT/docs"
echo "   📚 ReDoc: http://$BKKI_API_HOST:$BKKI_API_PORT/redoc"
echo "   🔐 Auth Token: POST http://$BKKI_API_HOST:$BKKI_API_PORT/auth/token"
echo "   💬 Chat: POST http://$BKKI_API_HOST:$BKKI_API_PORT/chat"

echo ""
echo "🔑 Test-Credentials:"
echo "   Username: bundeskanzler"
echo "   Password: ki2025"

echo ""
echo "🚀 Starte API-Server..."

# Starte Uvicorn Server
uvicorn bundeskanzler_api:app \
    --host "$BKKI_API_HOST" \
    --port "$BKKI_API_PORT" \
    --workers "$BKKI_API_WORKERS" \
    --log-level "${BKKI_LOG_LEVEL,,}" \
    --access-log \
    --reload \
    --reload-dir . \
    --reload-include "*.py" \
    --timeout-keep-alive 65 \
    --timeout-graceful-shutdown 30