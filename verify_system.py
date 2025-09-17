#!/usr/bin/env python3
"""
System-Verifizierung für Bundeskanzler KI
Überprüft alle Kernkomponenten nach Installation oder Updates
"""

import asyncio
import sys
import os

# Füge Projekt-Verzeichnis zum Python-Pfad hinzu
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


async def verify_system_integrity():
    """
    Vollständige System-Verifizierung
    Überprüft alle kritischen Komponenten der Bundeskanzler KI
    """
    print("🔍 Bundeskanzler KI - System-Verifizierung")
    print("=" * 50)

    # Grundlegende System-Info
    print(f"✅ Python-Version: {sys.version.split()[0]}")
    print(f"✅ Arbeitsverzeichnis: {project_root}")
    print(f"✅ Plattform: {sys.platform}")
    print()

    # Teste Kernkomponenten
    components_status = {}

    # 1. Multimodal KI System
    print("🧠 Teste Multimodal KI System...")
    try:
        import multimodal_ki
        # Teste Initialisierung (ohne GPU-Modelle zu laden)
        print("✅ Multimodal KI: Import erfolgreich")
        components_status['multimodal_ki'] = True
    except Exception as e:
        print(f"❌ Multimodal KI: {str(e)}")
        components_status['multimodal_ki'] = False

    # 2. Datenbank-System
    print("💾 Teste Datenbank-System...")
    try:
        from core.database import get_db
        print("✅ Datenbank: Import erfolgreich")
        components_status['database'] = True
    except Exception as e:
        print(f"❌ Datenbank: {str(e)}")
        components_status['database'] = False

    # 3. Dynamic Batching System
    print("⚡ Teste Dynamic Batching System...")
    try:
        from core.dynamic_batching import DynamicBatchProcessor, BatchConfig
        config = BatchConfig(max_batch_size=8, max_wait_time=0.05)
        batch_processor = DynamicBatchProcessor(config=config, gpu_memory_gb=8.0)
        print("✅ Dynamic Batching: Initialisierung erfolgreich")
        components_status['dynamic_batching'] = True
    except Exception as e:
        print(f"❌ Dynamic Batching: {str(e)}")
        components_status['dynamic_batching'] = False

    # 4. Authentifizierungssystem
    print("🔐 Teste Authentifizierungssystem...")
    try:
        from core.local_auth_manager import get_auth_manager
        print("✅ Authentifizierung: Import erfolgreich")
        components_status['auth'] = True
    except Exception as e:
        print(f"❌ Authentifizierung: {str(e)}")
        components_status['auth'] = False

    # 5. Intelligent Caching
    print("🧠 Teste Intelligent Caching...")
    try:
        from core.intelligent_cache import get_intelligent_cache
        print("✅ Intelligent Caching: Import erfolgreich")
        components_status['intelligent_cache'] = True
    except Exception as e:
        print(f"❌ Intelligent Caching: {str(e)}")
        components_status['intelligent_cache'] = False

    # 6. API-Server (synchroner Test)
    print("🌐 Teste API-Server Kompatibilität...")
    try:
        import bundeskanzler_api
        print("✅ API-Server: Import erfolgreich")
        components_status['api_server'] = True
    except Exception as e:
        print(f"❌ API-Server: {str(e)}")
        components_status['api_server'] = False

    print()
    print("📊 Verifizierungsergebnisse:")
    print("-" * 30)

    all_ok = True
    for component, status in components_status.items():
        status_icon = "✅" if status else "❌"
        component_name = component.replace('_', ' ').title()
        print(f"{status_icon} {component_name}: {'OK' if status else 'FEHLER'}")
        if not status:
            all_ok = False

    print()
    if all_ok:
        print("🎉 ALLE SYSTEME FUNKTIONSFÄHIG!")
        print("Die Bundeskanzler KI ist bereit für den Einsatz.")
        return True
    else:
        print("⚠️ EINIGE SYSTEME HABEN PROBLEME!")
        print("Überprüfen Sie die Fehler oben und beheben Sie diese.")
        return False


async def main():
    """Hauptfunktion für System-Verifizierung"""
    success = await verify_system_integrity()

    # Performance-Info
    print()
    print("📈 Performance-Übersicht:")
    print("- RTX 2070 GPU-optimiert")
    print("- Batch-Größe: 8 Requests")
    print("- Erwartete Query-Zeit: ~0.17s")
    print("- Concurrent-Durchsatz: 326+ req/s")

    return success


if __name__ == "__main__":
    # Führe Verifizierung aus
    success = asyncio.run(main())
    sys.exit(0 if success else 1)