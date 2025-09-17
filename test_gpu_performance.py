#!/usr/bin/env python3
"""
🚀 GPU Performance Test für RTX 2070 Optimierungen
===========================================

Testet die neuen Performance-Verbesserungen:
- Advanced Memory Management
- Dynamic Batch Adjustment
- Memory Pool Optimization
- Performance Monitoring
"""

import sys
import time
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core.gpu_manager import RTX2070Manager, AdvancedGPUMemoryManager
    print("✅ GPU Manager importiert")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    sys.exit(1)


def test_basic_gpu_functionality():
    """Testet grundlegende GPU-Funktionalität"""
    print("\n🔧 Teste grundlegende GPU-Funktionalität...")

    manager = RTX2070Manager()

    if not manager.is_gpu_available():
        print("❌ GPU nicht verfügbar - überspringe Tests")
        return False

    # GPU Stats abrufen
    stats = manager.get_gpu_stats()
    if stats:
        print("✅ GPU Stats erfolgreich abgerufen:")
        print(f"   💾 Memory: {stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f} GB")
        print(f"   🔥 GPU Util: {stats.gpu_utilization:.1f}%")
        print(f"   🌡️ Temp: {stats.temperature_c}°C")
    else:
        print("⚠️ GPU Stats nicht verfügbar")

    # Memory Summary
    mem_summary = manager.get_memory_summary()
    print(f"✅ Memory Summary: {mem_summary.get('status', 'UNKNOWN')}")

    return True


def test_advanced_features():
    """Testet Advanced Features"""
    print("\n🧠 Teste Advanced Features...")

    manager = RTX2070Manager()

    # Auto Memory Management testen
    print("   📊 Führe Auto Memory Management aus...")
    manager.auto_memory_management()

    # Performance Empfehlungen
    recommendations = manager.get_performance_recommendations()
    print("   💡 Performance-Empfehlungen:")
    for rec in recommendations:
        print(f"      {rec}")

    # Advanced Stats
    advanced_stats = manager.get_advanced_stats()
    print("   📈 Advanced Stats:")
    print(f"      Memory Efficiency: {advanced_stats.get('memory_efficiency', 0):.2f}")
    print(f"      Fragmentation: {advanced_stats.get('fragmentation_ratio', 0):.2f}")
    print(f"      Memory Pools: {advanced_stats.get('active_memory_pools', 0)}")

    return True


def main():
    """Hauptfunktion für GPU Performance Tests"""
    print("🚀 RTX 2070 Performance Test Suite")
    print("=" * 40)

    # Tests ausführen
    test1 = test_basic_gpu_functionality()
    test2 = test_advanced_features()

    # Zusammenfassung
    passed = sum([test1, test2])
    total = 2

    print(f"\n📊 Ergebnis: {passed}/{total} Tests erfolgreich")

    if passed == total:
        print("🎉 Alle Tests erfolgreich! RTX 2070 Performance-Optimierungen aktiv.")
    else:
        print("⚠️ Einige Tests fehlgeschlagen.")


if __name__ == "__main__":
    main()