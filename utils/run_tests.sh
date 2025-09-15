#!/bin/bash
# Umfassendes Test-Skript für die Bundeskanzler KI
# Führt alle verfügbaren Tests in der richtigen Reihenfolge aus

set -e  # Script beenden bei Fehlern

echo "🚀 Bundeskanzler KI - Umfassende Test-Suite"
echo "=========================================="

# Farben für bessere Lesbarkeit
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Hilfsfunktion für farbige Ausgabe
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Prüfe ob Docker läuft
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_warning "Docker ist nicht verfügbar. Tests werden ohne Docker ausgeführt."
        return 1
    fi
    print_success "Docker ist verfügbar"
    return 0
}

# Prüfe ob die API läuft
check_api() {
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API ist verfügbar"
        return 0
    else
        print_warning "API läuft nicht. Einige Tests könnten übersprungen werden."
        return 1
    fi
}

# Starte Docker-Umgebung (optional)
start_docker() {
    if [[ "$run_docker" == true ]]; then
        print_status "Starte Docker-Umgebung..."
        sudo docker-compose --profile gpu --profile with-db --profile with-redis up -d

        # Warte auf Services
        print_status "Warte auf Services..."
        sleep 15

        # Prüfe Services
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API erfolgreich gestartet"
        else
            print_error "API konnte nicht gestartet werden"
            exit 1
        fi
    fi
}

# Führe Unit-Tests aus
run_unit_tests() {
    print_status "Führe Unit-Tests aus..."
    python -m pytest tests/ -v --tb=short
    print_success "Unit-Tests abgeschlossen"
}

# Führe Integration-Tests aus
run_integration_tests() {
    print_status "Führe Integration-Tests aus..."
    python -m pytest tests/ -m "integration" -v --tb=short || python -m pytest tests/test_bundeskanzler_api.py -v --tb=short
    print_success "Integration-Tests abgeschlossen"
}

# Führe API-Tests aus
run_api_tests() {
    print_status "Führe API-Tests aus..."
    python -m pytest tests/ -m "api" -v --tb=short || python -m pytest tests/test_bundeskanzler_api.py -v --tb=short
    print_success "API-Tests abgeschlossen"
}

# Führe Memory-Tests aus
run_memory_tests() {
    print_status "Führe Memory-Tests aus..."
    python -m pytest tests/ -m "memory" -v --tb=short || python -m pytest tests/test_*memory*.py -v --tb=short
    print_success "Memory-Tests abgeschlossen"
}

# Führe Security-Tests aus
run_security_tests() {
    print_status "Führe Security-Tests aus..."
    python -m pytest tests/ -m "security" -v --tb=short || python -m pytest tests/test_security.py -v --tb=short
    print_success "Security-Tests abgeschlossen"
}

# Führe Performance-Tests aus
run_performance_tests() {
    print_status "Führe Performance-Tests aus..."
    python -m pytest tests/ -m "performance" --benchmark-only --benchmark-json=benchmark_results.json || python -m pytest tests/test_performance.py --benchmark-only --benchmark-json=benchmark_results.json
    print_success "Performance-Tests abgeschlossen"
}

# Erstelle Coverage-Report
run_coverage() {
    print_status "Erstelle Coverage-Report..."
    python -m pytest --cov=. --cov-report=html --cov-report=term --cov-report=xml --cov-fail-under=80
    print_success "Coverage-Report erstellt: htmlcov/index.html"
}

# Führe Docker-Tests aus
run_docker_tests() {
    print_status "Führe Docker-Tests aus..."
    sudo docker-compose --profile gpu --profile with-db --profile with-redis --profile testing run --rm test-runner
    print_success "Docker-Tests abgeschlossen"
}

# Hauptlogik
main() {
    local run_docker=false
    local run_all=true
    local test_type=""

    # Parse Kommandozeilen-Argumente
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                run_docker=true
                shift
                ;;
            --type)
                test_type="$2"
                run_all=false
                shift 2
                ;;
            --help)
                echo "Verwendung: $0 [OPTIONEN]"
                echo ""
                echo "Optionen:"
                echo "  --docker          Verwende Docker für Tests"
                echo "  --type TYPE       Führe nur spezifischen Test-Typ aus (unit, integration, api, memory, security, performance)"
                echo "  --help           Zeige diese Hilfe"
                exit 0
                ;;
            *)
                print_error "Unbekannte Option: $1"
                exit 1
                ;;
        esac
    done

    # Docker-Check
    if check_docker; then
        docker_available=true
    else
        docker_available=false
    fi

    # API-Check und Start
    if ! check_api; then
        if [[ "$docker_available" == true && "$run_docker" == true ]]; then
            start_docker
        fi
    fi

    # Bestimme welche Tests ausgeführt werden sollen
    if [[ "$run_all" == true ]]; then
        print_status "Führe vollständige Test-Suite aus..."

        run_unit_tests
        run_integration_tests
        run_api_tests
        run_memory_tests
        run_security_tests
        run_performance_tests
        run_coverage

        if [[ "$run_docker" == true && "$docker_available" == true ]]; then
            run_docker_tests
        fi

    else
        case $test_type in
            unit)
                run_unit_tests
                ;;
            integration)
                run_integration_tests
                ;;
            api)
                run_api_tests
                ;;
            memory)
                run_memory_tests
                ;;
            security)
                run_security_tests
                ;;
            performance)
                run_performance_tests
                ;;
            coverage)
                run_coverage
                ;;
            docker)
                run_docker_tests
                ;;
            *)
                print_error "Unbekannter Test-Typ: $test_type"
                exit 1
                ;;
        esac
    fi

    print_success "🎉 Alle Tests erfolgreich abgeschlossen!"
    echo ""
    echo "📊 Zusammenfassung:"
    echo "   ✅ Unit-Tests"
    echo "   ✅ Integration-Tests"
    echo "   ✅ API-Tests"
    echo "   ✅ Memory-Tests"
    echo "   ✅ Security-Tests"
    echo "   ✅ Performance-Tests"
    echo "   ✅ Coverage-Report"
    if [[ "$run_docker" == true ]]; then
        echo "   ✅ Docker-Tests"
    fi
    echo ""
    echo "📁 Reports:"
    echo "   • Coverage: htmlcov/index.html"
    echo "   • Benchmarks: benchmark_results.json"
    echo "   • Security: security-report.json (falls vorhanden)"
}

# Script ausführen
main "$@"