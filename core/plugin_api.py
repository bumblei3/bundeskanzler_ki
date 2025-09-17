#!/usr/bin/env python3
"""
Plugin-API für Bundeskanzler KI
REST-API-Endpunkte für Plugin-Management und -Steuerung
"""

import json
import logging
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, Response
from core.plugin_system import get_plugin_manager, PluginManager

# Logger für Plugin-API
logger = logging.getLogger(__name__)

class PluginAPI:
    """
    REST-API für Plugin-Management

    Stellt Endpunkte für das Laden, Entladen, Konfigurieren und Überwachen von Plugins bereit.
    """

    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        self.plugin_manager = plugin_manager or get_plugin_manager()
        self.blueprint = Blueprint('plugin_api', __name__, url_prefix='/api/plugins')

        # Registriere API-Endpunkte
        self._register_routes()

    def _register_routes(self) -> None:
        """Registriert alle API-Routen"""

        @self.blueprint.route('/', methods=['GET'])
        def get_plugins():
            """Gibt alle verfügbaren Plugins zurück"""
            try:
                plugins_info = self.plugin_manager.get_plugin_info()
                discovered_plugins = self.plugin_manager.discover_plugins()

                return jsonify({
                    'status': 'success',
                    'data': {
                        'loaded_plugins': plugins_info,
                        'available_plugins': discovered_plugins,
                        'total_loaded': len(plugins_info),
                        'total_available': len(discovered_plugins)
                    }
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Abrufen der Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>', methods=['GET'])
        def get_plugin(plugin_name: str):
            """Gibt Informationen über ein spezifisches Plugin zurück"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} nicht gefunden'
                    }), 404

                return jsonify({
                    'status': 'success',
                    'data': {
                        'name': plugin_name,
                        'metadata': plugin.metadata.__dict__,
                        'config': plugin.get_config().__dict__,
                        'enabled': plugin.is_enabled(),
                        'type': plugin.__class__.__name__
                    }
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Abrufen des Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/load', methods=['POST'])
        def load_plugin(plugin_name: str):
            """Lädt ein Plugin"""
            try:
                config = request.get_json() or {}
                success = self.plugin_manager.load_plugin(plugin_name, config)

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'Plugin {plugin_name} erfolgreich geladen'
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} konnte nicht geladen werden'
                    }), 400
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Laden des Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/unload', methods=['POST'])
        def unload_plugin(plugin_name: str):
            """Entlädt ein Plugin"""
            try:
                success = self.plugin_manager.unload_plugin(plugin_name)

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'Plugin {plugin_name} erfolgreich entladen'
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} konnte nicht entladen werden'
                    }), 400
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Entladen des Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/config', methods=['GET'])
        def get_plugin_config(plugin_name: str):
            """Gibt die Konfiguration eines Plugins zurück"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} nicht gefunden'
                    }), 404

                return jsonify({
                    'status': 'success',
                    'data': plugin.get_config().__dict__
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Abrufen der Konfiguration: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/config', methods=['PUT'])
        def update_plugin_config(plugin_name: str):
            """Aktualisiert die Konfiguration eines Plugins"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} nicht gefunden'
                    }), 404

                config_data = request.get_json()
                if not config_data:
                    return jsonify({
                        'status': 'error',
                        'message': 'Keine Konfigurationsdaten bereitgestellt'
                    }), 400

                plugin.update_config(config_data)

                return jsonify({
                    'status': 'success',
                    'message': f'Konfiguration von Plugin {plugin_name} aktualisiert'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Aktualisieren der Konfiguration: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/enable', methods=['POST'])
        def enable_plugin(plugin_name: str):
            """Aktiviert ein Plugin"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} nicht gefunden'
                    }), 404

                plugin.enable()
                return jsonify({
                    'status': 'success',
                    'message': f'Plugin {plugin_name} aktiviert'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Aktivieren des Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/disable', methods=['POST'])
        def disable_plugin(plugin_name: str):
            """Deaktiviert ein Plugin"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} nicht gefunden'
                    }), 404

                plugin.disable()
                return jsonify({
                    'status': 'success',
                    'message': f'Plugin {plugin_name} deaktiviert'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Deaktivieren des Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/<plugin_name>/execute/<method_name>', methods=['POST'])
        def execute_plugin_method(plugin_name: str, method_name: str):
            """Führt eine Methode eines Plugins aus"""
            try:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if not plugin:
                    return jsonify({
                        'status': 'error',
                        'message': f'Plugin {plugin_name} nicht gefunden'
                    }), 404

                # Parameter aus der Anfrage extrahieren
                params = request.get_json() or {}
                args = params.get('args', [])
                kwargs = params.get('kwargs', {})

                # Methode sicher ausführen
                result = plugin.execute_safe(method_name, *args, **kwargs)

                return jsonify({
                    'status': 'success',
                    'data': result
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler bei der Ausführung: {str(e)}'
                }), 500

        @self.blueprint.route('/reload', methods=['POST'])
        def reload_plugins():
            """Lädt alle Plugins neu"""
            try:
                # Entlade alle Plugins
                unloaded = self.plugin_manager.unload_all_plugins()

                # Lade alle Plugins neu
                loaded = self.plugin_manager.load_all_plugins()

                return jsonify({
                    'status': 'success',
                    'message': f'{unloaded} Plugins entladen, {loaded} Plugins geladen'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Neuladen der Plugins: {str(e)}'
                }), 500

        @self.blueprint.route('/types', methods=['GET'])
        def get_plugin_types():
            """Gibt verfügbare Plugin-Typen zurück"""
            try:
                from core.plugin_system import (
                    TextProcessingPlugin, ImageProcessingPlugin,
                    AudioProcessingPlugin, HookPlugin
                )

                plugin_types = {
                    'TextProcessingPlugin': {
                        'description': 'Plugins zur Textverarbeitung und -verbesserung',
                        'methods': ['process_text']
                    },
                    'ImageProcessingPlugin': {
                        'description': 'Plugins zur Bildverarbeitung und -analyse',
                        'methods': ['process_image']
                    },
                    'AudioProcessingPlugin': {
                        'description': 'Plugins zur Audioverarbeitung und Transkription',
                        'methods': ['process_audio']
                    },
                    'HookPlugin': {
                        'description': 'Plugins mit System-Hooks für Lebenszyklus-Ereignisse',
                        'hooks': ['on_request_start', 'on_request_end', 'on_error', 'on_system_start', 'on_system_shutdown']
                    }
                }

                return jsonify({
                    'status': 'success',
                    'data': plugin_types
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Abrufen der Plugin-Typen: {str(e)}'
                }), 500

        @self.blueprint.route('/monitoring/metrics', methods=['GET'])
        def get_monitoring_metrics():
            """Gibt Monitoring-Metriken zurück (falls Monitoring-Plugin verfügbar)"""
            try:
                monitoring_plugin = self.plugin_manager.get_plugin('monitoring')
                if not monitoring_plugin:
                    return jsonify({
                        'status': 'error',
                        'message': 'Monitoring-Plugin nicht verfügbar'
                    }), 404

                # System-Metriken abrufen
                system_metrics = monitoring_plugin.get_system_metrics()
                plugin_metrics = monitoring_plugin.get_plugin_metrics()

                return jsonify({
                    'status': 'success',
                    'data': {
                        'system_metrics': system_metrics,
                        'plugin_metrics': plugin_metrics
                    }
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Abrufen der Metriken: {str(e)}'
                }), 500

        @self.blueprint.route('/security/report', methods=['GET'])
        def get_security_report():
            """Gibt Sicherheitsbericht zurück (falls Security-Plugin verfügbar)"""
            try:
                security_plugin = self.plugin_manager.get_plugin('security')
                if not security_plugin:
                    return jsonify({
                        'status': 'error',
                        'message': 'Security-Plugin nicht verfügbar'
                    }), 404

                # Sicherheitsbericht abrufen
                report = security_plugin.get_security_report()

                return jsonify({
                    'status': 'success',
                    'data': report
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Fehler beim Abrufen des Sicherheitsberichts: {str(e)}'
                }), 500

# Hilfsfunktionen für die Integration mit der Haupt-API
def create_plugin_blueprint() -> Blueprint:
    """
    Erstellt und konfiguriert das Plugin-API Blueprint

    Returns:
        Flask Blueprint für Plugin-API
    """
    api = PluginAPI()
    return api.blueprint

def register_plugin_api(app, plugin_manager: Optional[PluginManager] = None) -> None:
    """
    Registriert die Plugin-API bei einer Flask-Anwendung

    Args:
        app: Flask-Anwendung
        plugin_manager: Optionaler Plugin-Manager (sonst wird der globale verwendet)
    """
    blueprint = create_plugin_blueprint()
    app.register_blueprint(blueprint)
    logger.info("Plugin-API erfolgreich registriert")

def get_plugin_manager_for_api() -> PluginManager:
    """
    Gibt den Plugin-Manager für API-Zwecke zurück

    Returns:
        PluginManager-Instanz
    """
    return get_plugin_manager()