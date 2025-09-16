#!/bin/bash

# Bundeskanzler KI Docker Compose Start Script

echo "🚀 Bundeskanzler KI - Docker Compose Setup"
echo "=========================================="

# Funktion für farbige Ausgabe
print_info() {
    echo -e "\033[0;34mℹ️  $1\033[0m"
}

print_success() {
    echo -e "\033[0;32m✅ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠️  $1\033[0m"
}

print_error() {
    echo -e "\033[0;31m❌ $1\033[0m"
}

# Überprüfe ob docker-compose verfügbar ist
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose ist nicht installiert!"
    exit 1
fi

# Überprüfe ob .env Datei existiert
if [ ! -f ".env" ]; then
    print_warning ".env Datei nicht gefunden. Erstelle Standard-Konfiguration..."
    cp .env.example .env 2>/dev/null || print_warning "Keine .env.example gefunden. Verwende Standard-Werte."
fi

echo ""
echo "Verfügbare Profile:"
echo "  gpu           - API mit GPU-Unterstützung"
echo "  with-db       - Mit PostgreSQL Datenbank"
echo "  with-redis    - Mit Redis Caching"
echo "  with-monitoring - Mit Prometheus + Grafana"
echo ""

# Frage nach gewünschten Services
read -p "Welche Services möchten Sie starten? (gpu,with-db,with-redis,with-monitoring oder 'all' für alles): " services

case $services in
    "gpu")
        print_info "Starte API mit GPU-Unterstützung..."
        docker-compose --profile gpu up -d bundeskanzler-ki
        ;;
    "with-db")
        print_info "Starte API mit PostgreSQL Datenbank..."
        docker-compose --profile gpu --profile with-db up -d
        ;;
    "with-redis")
        print_info "Starte API mit Redis Caching..."
        docker-compose --profile gpu --profile with-redis up -d
        ;;
    "with-monitoring")
        print_info "Starte API mit Monitoring Stack..."
        docker-compose --profile gpu --profile with-monitoring up -d
        ;;
    "all")
        print_info "Starte alle Services (GPU, DB, Redis, Monitoring)..."
        docker-compose --profile gpu --profile with-db --profile with-redis --profile with-monitoring up -d
        ;;
    *)
        print_error "Ungültige Auswahl. Verwende Standard: GPU + DB + Redis"
        docker-compose --profile gpu --profile with-db --profile with-redis up -d
        ;;
esac

echo ""
print_info "Warte auf Service-Start..."
sleep 10

# Zeige Status
echo ""
print_info "Service-Status:"
docker-compose ps

echo ""
print_success "Bundeskanzler KI ist bereit!"
echo ""
echo "🌐 URLs:"
echo "  API:           http://localhost:8000"
echo "  API Docs:      http://localhost:8000/docs"
echo "  pgAdmin:       http://localhost:8080 (admin@bundeskanzler-ki.de / admin123!)"
echo "  Grafana:       http://localhost:3000 (admin / admin123!)"
echo "  Prometheus:    http://localhost:9090"
echo ""
echo "🛑 Zum Stoppen: docker-compose down"
echo "📊 Logs anzeigen: docker-compose logs -f"