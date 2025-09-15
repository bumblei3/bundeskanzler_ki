#!/bin/bash

# Docker Deployment Script für Bundeskanzler-KI
# Automatisiertes Deployment mit Health Checks und Rollback

set -e

# Configuration
PROJECT_NAME="bundeskanzler-ki"
IMAGE_TAG=${1:-"latest"}
ENVIRONMENT=${2:-"production"}
BACKUP_DIR="./backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Create backup of current data
create_backup() {
    log "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    tar -czf "$BACKUP_FILE" \
        --exclude='./logs/*' \
        --exclude='./reports/*' \
        --exclude='./__pycache__' \
        ./data ./models 2>/dev/null || true
    
    success "Backup created: $BACKUP_FILE"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose -f docker-compose.dev.yml build --no-cache
    else
        docker-compose -f docker-compose.prod.yml build --no-cache
    fi
    
    success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose -f docker-compose.dev.yml up -d
    else
        docker-compose -f docker-compose.prod.yml up -d
    fi
    
    success "Services deployed"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts"
        
        # Check main service
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            success "Main service is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Health check failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check other services
    if [ "$ENVIRONMENT" == "production" ]; then
        log "Checking Grafana..."
        if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
            success "Grafana is healthy"
        else
            warning "Grafana health check failed"
        fi
        
        log "Checking Prometheus..."
        if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
            success "Prometheus is healthy"
        else
            warning "Prometheus health check failed"
        fi
    fi
    
    success "Health checks completed"
}

# Rollback function
rollback() {
    error "Deployment failed, initiating rollback..."
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose -f docker-compose.dev.yml down
    else
        docker-compose -f docker-compose.prod.yml down
    fi
    
    # Restore from latest backup
    LATEST_BACKUP=$(ls -t $BACKUP_DIR/backup_*.tar.gz 2>/dev/null | head -n1)
    if [ -n "$LATEST_BACKUP" ]; then
        log "Restoring from backup: $LATEST_BACKUP"
        tar -xzf "$LATEST_BACKUP"
        success "Backup restored"
    fi
    
    error "Rollback completed"
    exit 1
}

# Update configurations
update_configs() {
    log "Updating configurations..."
    
    # Create monitoring directories
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/provisioning/dashboards
    mkdir -p monitoring/grafana/provisioning/datasources
    mkdir -p nginx/ssl
    
    # Copy configuration files if they don't exist
    if [ ! -f monitoring/prometheus.yml ]; then
        cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'bundeskanzler-ki'
    static_configs:
      - targets: ['bundeskanzler-ki:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    fi
    
    # Nginx configuration
    if [ ! -f nginx/nginx.conf ]; then
        cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream bundeskanzler_ki {
        server bundeskanzler-ki:8000;
    }
    
    upstream grafana {
        server grafana:3000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://bundeskanzler_ki;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
        
        location /grafana/ {
            proxy_pass http://grafana/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
    }
}
EOF
    fi
    
    success "Configurations updated"
}

# Cleanup old images and containers
cleanup() {
    log "Cleaning up old images and containers..."
    
    # Remove old containers
    docker container prune -f
    
    # Remove old images
    docker image prune -f
    
    # Remove unused volumes (be careful with this in production)
    if [ "$ENVIRONMENT" == "development" ]; then
        docker volume prune -f
    fi
    
    success "Cleanup completed"
}

# Show deployment info
show_info() {
    log "Deployment Information:"
    echo "========================"
    echo "Environment: $ENVIRONMENT"
    echo "Image Tag: $IMAGE_TAG"
    echo ""
    echo "Services:"
    if [ "$ENVIRONMENT" == "development" ]; then
        echo "  - Bundeskanzler-KI Dev: http://localhost:8000"
        echo "  - Jupyter Notebook: http://localhost:8888"
        echo "  - PostgreSQL: localhost:5432"
    else
        echo "  - Bundeskanzler-KI: http://localhost"
        echo "  - Grafana Dashboard: http://localhost:3000"
        echo "  - Prometheus: http://localhost:9090"
    fi
    echo ""
    success "Deployment completed successfully!"
}

# Show logs
show_logs() {
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose -f docker-compose.dev.yml logs -f
    else
        docker-compose -f docker-compose.prod.yml logs -f
    fi
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose -f docker-compose.dev.yml down
    else
        docker-compose -f docker-compose.prod.yml down
    fi
    
    success "Services stopped"
}

# Main deployment function
main() {
    case "${3:-deploy}" in
        "deploy")
            log "Starting deployment of Bundeskanzler-KI ($ENVIRONMENT)..."
            check_prerequisites
            create_backup
            update_configs
            build_images
            deploy_services
            
            if health_check; then
                show_info
            else
                rollback
            fi
            ;;
        "stop")
            stop_services
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 [image_tag] [environment] [action]"
            echo "  image_tag: Docker image tag (default: latest)"
            echo "  environment: production|development (default: production)"
            echo "  action: deploy|stop|logs|cleanup (default: deploy)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Deploy production with latest tag"
            echo "  $0 v2.0 production deploy    # Deploy production with v2.0 tag"
            echo "  $0 latest development deploy # Deploy development environment"
            echo "  $0 latest production stop    # Stop production services"
            echo "  $0 latest production logs    # Show production logs"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"