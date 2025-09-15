# Dockerfile für Bundeskanzler KI
FROM python:3.12-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libpq-dev \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY . .

# Test-Abhängigkeiten installieren (für Test-Container)
RUN pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark pytest-html pytest-xdist httpx locust responses freezegun

# Port freigeben
EXPOSE 8000

# Health-Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startbefehl
CMD ["uvicorn", "bundeskanzler_api:app", "--host", "0.0.0.0", "--port", "8000"]