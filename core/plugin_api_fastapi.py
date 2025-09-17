#!/usr/bin/env python3
"""
FastAPI Plugin-API für Bundeskanzler KI
REST-API-Endpunkte für Plugin-Management und -Steuerung mit FastAPI
"""

import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from core.plugin_system import get_plugin_manager, PluginManager

# Logger für Plugin-API
logger = logging.getLogger(__name__)

# Pydantic-Modelle für API
class PluginConfigModel(BaseModel):
    """Plugin-Konfiguration Modell"""
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    settings: Dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True

class PluginLoadRequest(BaseModel):
    """Plugin-Lade-Anfrage Modell"""
    config: Optional[PluginConfigModel] = None

class PluginExecuteRequest(BaseModel):
    """Plugin-Ausführungs-Anfrage Modell"""
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class PluginInfo(BaseModel):
    """Plugin-Info Modell"""
    name: str
    metadata: Dict[str, Any]
    config: Dict[str, Any]
    enabled: bool
    type: str

class PluginList(BaseModel):
    """Plugin-Liste Modell"""
    loaded_plugins: Dict[str, PluginInfo]
    available_plugins: List[str]
    total_loaded: int
    total_available: int

class PluginTypeInfo(BaseModel):
    """Plugin-Typ Info Modell"""
    description: str
    methods: Optional[List[str]] = None
    hooks: Optional[List[str]] = None

class PluginTypes(BaseModel):
    """Plugin-Typen Modell"""
    TextProcessingPlugin: PluginTypeInfo
    ImageProcessingPlugin: PluginTypeInfo
    AudioProcessingPlugin: PluginTypeInfo
    HookPlugin: PluginTypeInfo

class MonitoringMetrics(BaseModel):
    """Monitoring-Metriken Modell"""
    system_metrics: Dict[str, Any]
    plugin_metrics: Dict[str, Any]

class SecurityReport(BaseModel):
    """Sicherheitsbericht Modell"""
    total_blocked_requests: int
    total_suspicious_requests: int
    blocked_ips_count: int
    recent_events: List[Dict[str, Any]]
    risk_assessment: str
    recommendations: List[str]

class PluginAPIRouter:
    """
    FastAPI Router für Plugin-Management

    Stellt REST-API-Endpunkte für das Laden, Entladen, Konfigurieren und Überwachen von Plugins bereit.
    """

    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        self.plugin_manager = plugin_manager or get_plugin_manager()
        self.router = APIRouter(
            prefix="/api/plugins",
            tags=["plugins"],
            responses={404: {"description": "Plugin nicht gefunden"}}
        )

        # Registriere API-Endpunkte
        self._register_routes()

    def _register_routes(self) -> None:
        """Registriert alle API-Routen"""

        @self.router.get("/", response_model=PluginList)
        async def get_plugins():
            """Gibt alle verfügbaren Plugins zurück"""
            try:
                plugins_info = self.plugin_manager.get_plugin_info()
                discovered_plugins = self.plugin_manager.discover_plugins()

                # Konvertiere zu Pydantic-Modellen
                loaded_plugins = {}
                for name, info in plugins_info.items():
                    loaded_plugins[name] = PluginInfo(**info)

                return PluginList(
                    loaded_plugins=loaded_plugins,
                    available_plugins=discovered_plugins,
                    total_loaded=len(plugins_info),
                    total_available=len(discovered_plugins)
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Plugins: {str(e)}")

        @self.router.get("/{plugin_name}", response_model=PluginInfo)
        async def get_plugin(plugin_name: str):
            """Gibt Informationen über ein spezifisches Plugin zurück"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} nicht gefunden")

                return PluginInfo(
                    name=plugin_name,
                    metadata=plugin.metadata.__dict__,
                    config=plugin.get_config().__dict__,
                    enabled=plugin.is_enabled(),
                    type=plugin.__class__.__name__
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen des Plugins: {str(e)}")

        @self.router.post("/{plugin_name}/load")
        async def load_plugin(plugin_name: str, request: PluginLoadRequest = None):
            """Lädt ein Plugin"""
            try:
                config = request.config.__dict__ if request and request.config else None
                success = self.plugin_manager.load_plugin(plugin_name, config)

                if success:
                    return {"status": "success", "message": f"Plugin {plugin_name} erfolgreich geladen"}
                else:
                    raise HTTPException(status_code=400, detail=f"Plugin {plugin_name} konnte nicht geladen werden")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Plugins: {str(e)}")

        @self.router.post("/{plugin_name}/unload")
        async def unload_plugin(plugin_name: str):
            """Entlädt ein Plugin"""
            try:
                success = self.plugin_manager.unload_plugin(plugin_name)

                if success:
                    return {"status": "success", "message": f"Plugin {plugin_name} erfolgreich entladen"}
                else:
                    raise HTTPException(status_code=400, detail=f"Plugin {plugin_name} konnte nicht entladen werden")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Entladen des Plugins: {str(e)}")

        @self.router.get("/{plugin_name}/config")
        async def get_plugin_config(plugin_name: str):
            """Gibt die Konfiguration eines Plugins zurück"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} nicht gefunden")

                return plugin.get_config().__dict__
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Konfiguration: {str(e)}")

        @self.router.put("/{plugin_name}/config")
        async def update_plugin_config(plugin_name: str, config: Dict[str, Any]):
            """Aktualisiert die Konfiguration eines Plugins"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} nicht gefunden")

                plugin.update_config(config)
                return {"status": "success", "message": f"Konfiguration von Plugin {plugin_name} aktualisiert"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Aktualisieren der Konfiguration: {str(e)}")

        @self.router.post("/{plugin_name}/enable")
        async def enable_plugin(plugin_name: str):
            """Aktiviert ein Plugin"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} nicht gefunden")

                plugin.enable()
                return {"status": "success", "message": f"Plugin {plugin_name} aktiviert"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Aktivieren des Plugins: {str(e)}")

        @self.router.post("/{plugin_name}/disable")
        async def disable_plugin(plugin_name: str):
            """Deaktiviert ein Plugin"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} nicht gefunden")

                plugin.disable()
                return {"status": "success", "message": f"Plugin {plugin_name} deaktiviert"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Deaktivieren des Plugins: {str(e)}")

        @self.router.post("/{plugin_name}/execute/{method_name}")
        async def execute_plugin_method(plugin_name: str, method_name: str, request: PluginExecuteRequest = None):
            """Führt eine Methode eines Plugins aus"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} nicht gefunden")

                # Parameter aus der Anfrage extrahieren
                args = request.args if request else []
                kwargs = request.kwargs if request else {}

                # Methode sicher ausführen
                result = plugin.execute_safe(method_name, *args, **kwargs)

                return {"status": "success", "data": result}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler bei der Ausführung: {str(e)}")

        @self.router.post("/reload")
        async def reload_plugins():
            """Lädt alle Plugins neu"""
            try:
                # Entlade alle Plugins
                unloaded = self.plugin_manager.unload_all_plugins()

                # Lade alle Plugins neu
                loaded = self.plugin_manager.load_all_plugins()

                return {
                    "status": "success",
                    "message": f"{unloaded} Plugins entladen, {loaded} Plugins geladen"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Neuladen der Plugins: {str(e)}")

        @self.router.get("/types", response_model=PluginTypes)
        async def get_plugin_types():
            """Gibt verfügbare Plugin-Typen zurück"""
            try:
                plugin_types = PluginTypes(
                    TextProcessingPlugin=PluginTypeInfo(
                        description="Plugins zur Textverarbeitung und -verbesserung",
                        methods=["process_text"]
                    ),
                    ImageProcessingPlugin=PluginTypeInfo(
                        description="Plugins zur Bildverarbeitung und -analyse",
                        methods=["process_image"]
                    ),
                    AudioProcessingPlugin=PluginTypeInfo(
                        description="Plugins zur Audioverarbeitung und Transkription",
                        methods=["process_audio"]
                    ),
                    HookPlugin=PluginTypeInfo(
                        description="Plugins mit System-Hooks für Lebenszyklus-Ereignisse",
                        hooks=["on_request_start", "on_request_end", "on_error", "on_system_start", "on_system_shutdown"]
                    )
                )

                return plugin_types
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Plugin-Typen: {str(e)}")

        @self.router.get("/monitoring/metrics", response_model=MonitoringMetrics)
        async def get_monitoring_metrics():
            """Gibt Monitoring-Metriken zurück (falls Monitoring-Plugin verfügbar)"""
            try:
                monitoring_plugin = self.plugin_manager.get_plugin('monitoring')
                if not monitoring_plugin:
                    raise HTTPException(status_code=404, detail="Monitoring-Plugin nicht verfügbar")

                # System-Metriken abrufen
                system_metrics = monitoring_plugin.get_system_metrics()
                plugin_metrics = monitoring_plugin.get_plugin_metrics()

                return MonitoringMetrics(
                    system_metrics=system_metrics,
                    plugin_metrics=plugin_metrics
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Metriken: {str(e)}")

        @self.router.get("/security/report", response_model=SecurityReport)
        async def get_security_report():
            """Gibt Sicherheitsbericht zurück (falls Security-Plugin verfügbar)"""
            try:
                security_plugin = self.plugin_manager.get_plugin('security')
                if not security_plugin:
                    raise HTTPException(status_code=404, detail="Security-Plugin nicht verfügbar")

                # Sicherheitsbericht abrufen
                report = security_plugin.get_security_report()

                return SecurityReport(**report)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen des Sicherheitsberichts: {str(e)}")

# Hilfsfunktionen für die Integration mit FastAPI
def create_plugin_router() -> APIRouter:
    """
    Erstellt und konfiguriert den Plugin-API Router

    Returns:
        FastAPI APIRouter für Plugin-API
    """
    api = PluginAPIRouter()
    return api.router

def register_plugin_api_fastapi(app, plugin_manager: Optional[PluginManager] = None) -> None:
    """
    Registriert die Plugin-API bei einer FastAPI-Anwendung

    Args:
        app: FastAPI-Anwendung
        plugin_manager: Optionaler Plugin-Manager (sonst wird der globale verwendet)
    """
    router = create_plugin_router()
    app.include_router(router)
    logger.info("Plugin-API erfolgreich bei FastAPI registriert")

def get_plugin_manager_dependency() -> PluginManager:
    """
    Dependency Injection für Plugin-Manager

    Returns:
        PluginManager-Instanz
    """
    return get_plugin_manager()