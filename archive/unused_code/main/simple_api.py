#!/usr/bin/env python3
"""
Einfache FastAPI für Bundeskanzler-KI
Für Tests und Entwicklung
"""

import os
import sys
import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pfad zur RTX 2070 KI hinzufügen
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from core.rtx2070_bundeskanzler_ki import RTX2070BundeskanzlerKI

    KI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RTX 2070 KI nicht verfügbar: {e}")
    KI_AVAILABLE = False

app = FastAPI(
    title="Bundeskanzler KI API",
    description="Einfache API für die Bundeskanzler-KI",
    version="1.0.0",
)

# CORS für Web-GUI (produktionsbereit konfigurieren)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8502", "http://127.0.0.1:8502"],  # Nur lokale Entwicklung
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "de"
    fact_check: Optional[bool] = True

    @validator('query')
    def query_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('Query darf nicht leer sein')
        if len(v.strip()) > 2000:  # Begrenze Query-Länge
            raise ValueError('Query zu lang (max. 2000 Zeichen)')
        return v.strip()

    @validator('language')
    def language_must_be_valid(cls, v):
        valid_languages = ['de', 'en', 'fr', 'es', 'it']
        if v not in valid_languages:
            raise ValueError(f'Ungültige Sprache. Erlaubt: {valid_languages}')
        return v


class QueryResponse(BaseModel):
    response: str
    fact_check: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    sources: Optional[list] = None
    error: Optional[str] = None


# Globale KI-Instanz
ki_instance = None


@app.on_event("startup")
async def startup_event():
    """Initialisiere KI beim Start"""
    global ki_instance
    if KI_AVAILABLE:
        try:
            ki_instance = RTX2070BundeskanzlerKI()
            print("✅ RTX 2070 Bundeskanzler-KI erfolgreich initialisiert")
        except Exception as e:
            print(f"❌ Fehler bei KI-Initialisierung: {e}")
            ki_instance = None
    else:
        print("❌ KI-Module nicht verfügbar")


@app.get("/")
async def root():
    """Root-Endpunkt"""
    return {
        "message": "Bundeskanzler KI API",
        "status": "running",
        "ki_available": KI_AVAILABLE,
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health-Check"""
    return {"status": "healthy", "ki_initialized": ki_instance is not None}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Haupt-Query-Endpunkt"""
    start_time = time.time()
    client_ip = "unknown"  # In Produktion: request.client.host

    logger.info(f"Query von {client_ip}: {request.query[:100]}... (lang={request.language})")

    if not ki_instance:
        logger.error("KI nicht verfügbar")
        raise HTTPException(status_code=503, detail="KI nicht verfügbar")

    try:
        # Timeout für KI-Abfrage (30 Sekunden)
        import asyncio
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ki_instance.process_query_with_fact_check(query=request.query)
        )

        processing_time = time.time() - start_time
        logger.info(".2f")

        # Antwort formatieren
        response_text = result.get("response", "Keine Antwort verfügbar")

        # Fact-Check Daten extrahieren
        fact_check_data = None
        confidence = None
        sources = None

        if result.get("fact_check"):
            fact_check_data = result["fact_check"]
            confidence = result.get("confidence_score", 0.0)
            sources_count = result.get("sources_cited", 0)
            sources = [f"Quelle {i+1}" for i in range(sources_count)] if sources_count > 0 else []

        return QueryResponse(
            response=response_text,
            fact_check=fact_check_data,
            confidence=confidence,
            sources=sources,
            error=None,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(".2f")
        # Keine internen Fehler an Client weitergeben
        return QueryResponse(
            response="Entschuldigung, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage.",
            error="Interner Serverfehler",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
