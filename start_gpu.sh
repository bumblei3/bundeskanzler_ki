#!/bin/bash
# GPU-optimiertes Start-Script für Bundeskanzler-KI

echo "🚀 Starte Bundeskanzler-KI mit GPU-Unterstützung..."
echo "=================================================="

# Aktiviere virtuelle Umgebung
source bin/activate

# Lade GPU-Konfiguration
if [ -f .env ]; then
    source .env
    echo "✅ GPU-Konfiguration geladen"
else
    echo "⚠️ .env Datei nicht gefunden, verwende Standard-Konfiguration"
fi

# Zeige GPU-Status
echo ""
echo "📊 GPU-Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits

# Starte KI
echo ""
echo "🎯 Starte KI..."
python core/bundeskanzler_ki.py "$@"
