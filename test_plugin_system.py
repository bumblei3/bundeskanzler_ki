#!/usr/bin/env python3
"""
Test-Skript für das Plugin-System der Bundeskanzler KI
Testet die Kernfunktionalität des modularen Plugin-Systems
"""

import sys
import os
import time
import logging
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_plugin_system():
    """Testet das Plugin-System"""
    logger.info("=== Plugin-System-Test gestartet ===")

    try:
        from core.plugin_system import (
            get_plugin_manager, initialize_plugin_system, shutdown_plugin_system,
            TextProcessingPlugin, PluginMetadata
        )

        # Plugin-Manager initialisieren
        logger.info("1. Plugin-Manager initialisieren...")
        manager = get_plugin_manager()
        logger.info(f"Plugin-Manager erstellt: {manager}")

        # Verfügbare Plugins entdecken
        logger.info("2. Verfügbare Plugins entdecken...")
        discovered_plugins = manager.discover_plugins()
        logger.info(f"Entdeckte Plugins: {discovered_plugins}")

        # Plugins laden
        logger.info("3. Plugins laden...")
        loaded_count = manager.load_all_plugins()
        logger.info(f"{loaded_count} Plugins geladen")

        # Plugin-Informationen abrufen
        logger.info("4. Plugin-Informationen abrufen...")
        plugin_info = manager.get_plugin_info()
        for name, info in plugin_info.items():
            logger.info(f"Plugin {name}: {info['type']} - Aktiviert: {info['enabled']}")

        # Text-Verbesserungs-Plugin testen
        logger.info("5. Text-Verbesserungs-Plugin testen...")
        text_plugin = manager.get_plugin('text_improvement')
        if text_plugin:
            test_text = "Das ist ein test text mit fehlern. ist das richtig?"
            logger.info(f"Original-Text: {test_text}")

            improved_text = text_plugin.process_text(test_text)
            logger.info(f"Verbesserter Text: {improved_text}")
        else:
            logger.warning("Text-Verbesserungs-Plugin nicht gefunden")

        # Monitoring-Plugin testen
        logger.info("6. Monitoring-Plugin testen...")
        monitoring_plugin = manager.get_plugin('monitoring')
        if monitoring_plugin:
            system_metrics = monitoring_plugin.get_system_metrics()
            logger.info(f"System-Metriken: CPU {system_metrics['cpu_percent']:.1f}%, "
                       f"Speicher {system_metrics['memory_percent']:.1f}%")

            plugin_metrics = monitoring_plugin.get_plugin_metrics()
            logger.info(f"Plugin-Metriken: {len(plugin_metrics)} Plugins überwacht")
        else:
            logger.warning("Monitoring-Plugin nicht gefunden")

        # Sicherheits-Plugin testen
        logger.info("7. Sicherheits-Plugin testen...")
        security_plugin = manager.get_plugin('security')
        if security_plugin:
            # Test-Anfrage simulieren
            test_request = {
                'client_ip': '127.0.0.1',
                'content': 'SELECT * FROM users; DROP TABLE users;',
                'type': 'text'
            }

            security_result = security_plugin.check_request_security(test_request)
            logger.info(f"Sicherheitsprüfung: {security_result['risk_level']} Risiko")
            if security_result['issues']:
                logger.info(f"Sicherheitsprobleme: {security_result['issues']}")
        else:
            logger.warning("Sicherheits-Plugin nicht gefunden")

        # Plugin-Hooks testen
        logger.info("8. Plugin-Hooks testen...")
        test_request_data = {'type': 'test', 'content': 'Test-Anfrage'}

        # Hook für Request-Start
        manager.execute_hook('on_request_start', test_request_data)
        logger.info("Hook 'on_request_start' ausgeführt")

        # Hook für Request-Ende
        test_response = {'status': 'success', 'result': 'Test-Ergebnis'}
        manager.execute_hook('on_request_end', test_request_data, test_response)
        logger.info("Hook 'on_request_end' ausgeführt")

        # Plugins entladen
        logger.info("9. Plugins entladen...")
        unloaded_count = manager.unload_all_plugins()
        logger.info(f"{unloaded_count} Plugins entladen")

        logger.info("=== Plugin-System-Test erfolgreich abgeschlossen ===")
        return True

    except Exception as e:
        logger.error(f"Plugin-System-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plugin_api():
    """Testet die Plugin-API"""
    logger.info("=== Plugin-API-Test gestartet ===")

    try:
        from core.plugin_api import PluginAPI
        from core.plugin_system import get_plugin_manager

        # Plugin-API initialisieren
        logger.info("1. Plugin-API initialisieren...")
        manager = get_plugin_manager()
        api = PluginAPI(manager)
        logger.info("Plugin-API erstellt")

        # Blueprint testen
        logger.info("2. API-Blueprint testen...")
        blueprint = api.blueprint
        logger.info(f"Blueprint erstellt: {blueprint.name}")

        # API-Endpunkte auflisten
        logger.info("3. API-Endpunkte:")
        for rule in blueprint.deferred_functions:
            logger.info(f"  - {rule}")

        logger.info("=== Plugin-API-Test erfolgreich abgeschlossen ===")
        return True

    except Exception as e:
        logger.error(f"Plugin-API-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plugin_integration():
    """Testet die Plugin-Integration"""
    logger.info("=== Plugin-Integration-Test gestartet ===")

    try:
        from core.plugin_integration import get_plugin_integration

        # Plugin-Integration initialisieren
        logger.info("1. Plugin-Integration initialisieren...")
        integration = get_plugin_integration()
        logger.info("Plugin-Integration erstellt")

        # Integration initialisieren (ohne echte Systeme)
        logger.info("2. Integration initialisieren...")
        integration.initialize_integration()
        logger.info("Integration initialisiert")

        # Integration beenden
        logger.info("3. Integration beenden...")
        integration.shutdown_integration()
        logger.info("Integration beendet")

        logger.info("=== Plugin-Integration-Test erfolgreich abgeschlossen ===")
        return True

    except Exception as e:
        logger.error(f"Plugin-Integration-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hauptfunktion für Plugin-System-Tests"""
    logger.info("Starte umfassende Plugin-System-Tests...")

    # Plugin-Verzeichnis prüfen
    plugin_dir = Path(__file__).parent / "plugins"
    if not plugin_dir.exists():
        logger.error(f"Plugin-Verzeichnis nicht gefunden: {plugin_dir}")
        return False

    logger.info(f"Plugin-Verzeichnis: {plugin_dir}")

    # Tests ausführen
    tests = [
        ("Plugin-System", test_plugin_system),
        ("Plugin-API", test_plugin_api),
        ("Plugin-Integration", test_plugin_integration)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starte {test_name}-Test...")
        logger.info(f"{'='*50}")

        try:
            result = test_func()
            results.append((test_name, result))
            status = "ERFOLGREICH" if result else "FEHLGESCHLAGEN"
            logger.info(f"{test_name}-Test: {status}")
        except Exception as e:
            logger.error(f"{test_name}-Test unerwartet fehlgeschlagen: {e}")
            results.append((test_name, False))

    # Zusammenfassung
    logger.info(f"\n{'='*50}")
    logger.info("TEST-ZUSAMMENFASSUNG")
    logger.info(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓" if result else "✗"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1

    logger.info(f"\nErgebnis: {passed}/{total} Tests bestanden")

    if passed == total:
        logger.info("🎉 Alle Tests erfolgreich! Plugin-System ist bereit für den Einsatz.")
        return True
    else:
        logger.error("❌ Einige Tests sind fehlgeschlagen. Bitte überprüfen Sie die Implementierung.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)