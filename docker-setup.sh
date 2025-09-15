#!/bin/bash

# Docker Setup Script für Bundeskanzler KI

set -e

echo "🐳 Bundeskanzler KI - Docker Setup"
echo "=================================="

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funktion für farbigen Output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Überprüfe Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker ist nicht installiert. Bitte installiere Docker zuerst."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose ist nicht installiert. Bitte installiere docker-compose."
    exit 1
fi

# Überprüfe GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU erkannt - verwende GPU-Unterstützung"
    COMPOSE_FILE="docker-compose.yml"
    GPU_AVAILABLE=true
else
    print_warning "Keine NVIDIA GPU gefunden - verwende CPU-Modus"
    COMPOSE_FILE="docker-compose.cpu.yml"
    GPU_AVAILABLE=false
fi

# Erstelle notwendige Verzeichnisse
mkdir -p api_memory

# Build das Image
echo ""
echo "🔨 Baue Docker-Image..."
docker-compose -f $COMPOSE_FILE build

if [ $? -eq 0 ]; then
    print_status "Docker-Image erfolgreich gebaut"
else
    print_error "Fehler beim Bauen des Docker-Images"
    exit 1
fi

# Starte die Services
echo ""
echo "🚀 Starte Bundeskanzler KI..."
docker-compose -f $COMPOSE_FILE up -d

if [ $? -eq 0 ]; then
    print_status "Bundeskanzler KI erfolgreich gestartet"
else
    print_error "Fehler beim Starten der Services"
    exit 1
fi

# Warte auf Health-Check
echo ""
echo "⏳ Warte auf Anwendung (Health-Check)..."
sleep 10

# Überprüfe Health-Status
echo "🔍 Überprüfe Anwendungsstatus..."
if curl -f http://localhost:8000/health &> /dev/null; then
    print_status "Bundeskanzler KI ist bereit!"
    echo ""
    echo "🌐 API verfügbar unter: http://localhost:8000"
    echo "📚 Dokumentation: http://localhost:8000/docs"
    echo "🛑 Zum Stoppen: docker-compose down"
    echo "📊 Logs anzeigen: docker-compose logs -f"
else
    print_warning "Health-Check fehlgeschlagen. Überprüfe Logs:"
    echo "docker-compose logs bundeskanzler-ki"
fi

echo ""
print_status "Setup abgeschlossen!"