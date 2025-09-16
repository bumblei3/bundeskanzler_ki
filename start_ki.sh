#!/bin/bash
# Bundeskanzler-KI Starter Script
# Vereinfacht den Start der verschiedenen KI-Versionen

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}🤖 Bundeskanzler-KI Starter${NC}"
echo "=========================="

# Prüfe Virtual Environment
if [[ "$VIRTUAL_ENV" != *"bkki_venv"* ]]; then
    echo -e "${YELLOW}⚠️  Virtual Environment nicht aktiv. Aktiviere...${NC}"
    if [[ -f "bin/activate" ]]; then
        source bin/activate
        echo -e "${GREEN}✅ Virtual Environment aktiviert${NC}"
    else
        echo -e "${RED}❌ Virtual Environment nicht gefunden!${NC}"
        echo -e "${YELLOW}Führen Sie 'source bin/activate' manuell aus${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual Environment bereits aktiv${NC}"
fi

# Funktion: Verfügbare Optionen anzeigen
show_options() {
    echo -e "\n${GREEN}Verfügbare Optionen:${NC}"
    echo "1) 🎯 Verbesserte KI (EMPFOHLEN) - Beste Performance"
    echo "2) 🌐 Web-Interface (Streamlit)"
    echo "3) 📡 API Server"
    echo "4) 🔧 Original KI (Interaktiv)"
    echo "5) 🧪 Einfache KI (Test)"
    echo "6) 📊 Status & Logs"
    echo "7) 📡 Daten aktualisieren"
    echo "8) 🧹 Cache bereinigen"
    echo "9) ❌ Beenden"
    echo ""
}

# Funktion: KI-Status prüfen
check_status() {
    echo -e "${BLUE}📊 System Status:${NC}"
    
    # GPU Check
    if python -c "import tensorflow as tf; print('GPU verfügbar:', len(tf.config.list_physical_devices('GPU')) > 0)" 2>/dev/null; then
        echo -e "✅ TensorFlow GPU-Support: ${GREEN}OK${NC}"
    else
        echo -e "⚠️  TensorFlow GPU-Support: ${YELLOW}Nicht verfügbar${NC}"
    fi
    
    # RAG System Check
    if python -c "
import sys, os
sys.path.append('.')
from core.rag_system import RAGSystem
corpus_path = os.path.join('data', 'corpus.json')
RAGSystem(corpus_path=corpus_path)
print('RAG-System OK')
" &>/dev/null; then
        echo -e "✅ RAG-System: ${GREEN}OK${NC}"
    else
        echo -e "❌ RAG-System: ${RED}Fehler${NC}"
    fi
    
    # Models Check
    if [[ -f "models/rag_index.faiss" ]]; then
        echo -e "✅ RAG-Index: ${GREEN}Vorhanden${NC}"
    else
        echo -e "⚠️  RAG-Index: ${YELLOW}Nicht gefunden${NC}"
    fi
    
    # Logs
    if [[ -f "data/log.txt" ]]; then
        echo -e "📝 Letzte Log-Einträge:"
        tail -n 3 data/log.txt 2>/dev/null || echo "Keine Logs verfügbar"
    fi
}

# Funktion: Daten aktualisieren
update_data() {
    echo -e "${BLUE}📡 Aktualisiere Wissensbasis...${NC}"
    if python core/data_updater.py; then
        echo -e "${GREEN}✅ Wissensbasis aktualisiert${NC}"
    else
        echo -e "${RED}❌ Fehler bei der Datenaktualisierung${NC}"
    fi
}

# Hauptmenü
while true; do
    show_options
    read -p "Wählen Sie eine Option (1-9): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}🎯 Starte verbesserte KI...${NC}"
            python core/verbesserte_ki.py
            ;;
        2)
            echo -e "${GREEN}🌐 Starte Web-Interface...${NC}"
            echo -e "${BLUE}Öffne http://localhost:8501 in Ihrem Browser${NC}"
            python web/webgui_ki.py
            ;;
        3)
            echo -e "${GREEN}📡 Starte API Server...${NC}"
            echo -e "${BLUE}API verfügbar unter http://localhost:8000${NC}"
            python core/bundeskanzler_api.py
            ;;
        4)
            echo -e "${GREEN}🔧 Starte Original KI...${NC}"
            python core/bundeskanzler_ki.py interactive
            ;;
        5)
            echo -e "${YELLOW}🧪 Einfache KI (ARCHIVIERT)${NC}"
            echo -e "${BLUE}Die einfache KI wurde archiviert: archive/unused_code/ki_versions/${NC}"
            echo -e "${BLUE}Verwenden Sie stattdessen Option 1 (Verbesserte KI)${NC}"
            ;;
        6)
            check_status
            ;;
        7)
            update_data
            ;;
        8)
            clean_cache
            ;;
        9)
            echo -e "${GREEN}👋 Auf Wiedersehen!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Ungültige Option. Bitte 1-9 wählen.${NC}"
            ;;
    esac
    
    echo ""
    read -p "Drücken Sie Enter um fortzufahren..."
    clear
done