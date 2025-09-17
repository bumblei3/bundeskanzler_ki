#!/bin/bash
# Automatisches Import-Script für Bundeskanzler-KI
# Führt automatische Daten-Updates beim API-Start aus

echo "🔄 Starte automatischen Import-Prozess..."
echo "=========================================="

# Aktiviere virtuelle Umgebung
cd "$(dirname "$0")"
source bin/activate

# Prüfe auf verfügbare Updates
echo "📊 Prüfe auf Daten-Updates..."

# 1. Corpus-Validierung
if [ -f "corpus_validator.py" ]; then
    echo "🔍 Validiere Corpus..."
    python corpus_validator.py --quiet
fi

# 2. Datenbank-Updates (falls verfügbar)
if [ -f "init-db.sql" ]; then
    echo "🗄️ Prüfe Datenbank-Updates..."
    # Hier könnten DB-Updates implementiert werden
fi

# 3. Cache-Bereinigung (falls nötig)
if [ -d "tensorrt_cache" ]; then
    echo "🧹 Bereinige alten TensorRT-Cache..."
    find tensorrt_cache -name "*.cache" -mtime +7 -delete 2>/dev/null || true
fi

# 4. Log-Rotation
if [ -d "logs" ]; then
    echo "📝 Rotiere Logs..."
    find logs -name "*.log" -mtime +30 -delete 2>/dev/null || true
fi

echo "✅ Automatischer Import abgeschlossen"
echo "====================================="