#!/usr/bin/env python3
"""
Einfache Bundeskanzler-KI API für Tests
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App erstellen
app = FastAPI(
    title="Bundeskanzler-KI API",
    description="Einfache API für Bundeskanzler-KI Tests",
    version="1.0.0"
)

# CORS aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health-Check Endpunkt"""
    return {
        "status": "healthy",
        "message": "Bundeskanzler-KI API ist verfügbar",
        "version": "1.0.0"
    }

@app.post("/query")
async def query_endpoint(query: str = "Testfrage"):
    """Query-Endpunkt für KI-Anfragen"""
    return {
        "query": query,
        "response": f"Antwort auf: {query}",
        "status": "success"
    }

@app.get("/")
async def root():
    """Root-Endpunkt"""
    return {
        "message": "Bundeskanzler-KI API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    logger.info("🚀 Starte Bundeskanzler-KI API auf http://localhost:8000")
    logger.info("📖 OpenAPI-Dokumentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)