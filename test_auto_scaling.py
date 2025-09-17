#!/usr/bin/env python3
"""
Test für Auto-Scaling System Integration
Testet die Integration des Auto-Scaling Systems mit der multimodalen KI
"""

import sys
import os
import time
import json
from datetime import datetime

# Füge das Projektverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, '/home/tobber/bkki_venv')
sys.path.insert(0, '/home/tobber/bkki_venv/core')

try:
    from multimodal_ki import MultimodalTransformerModel
    from auto_scaling import AutoScaler, SystemMetrics, ScalingDecision
    print("✅ Import erfolgreich")
except ImportError as e:
    print(f"❌ Import fehlgeschlagen: {e}")
    sys.exit(1)

def test_auto_scaling_integration():
    """Testet die Auto-Scaling Integration"""
    print("\n🧪 Teste Auto-Scaling Integration...")

    try:
        # Erstelle MultimodalTransformerModel Instanz
        model = MultimodalTransformerModel()
        print("✅ MultimodalTransformerModel erstellt")

        # Teste System-Status
        status = model.get_system_status()
        print(f"✅ System-Status abgerufen: {status.get('auto_scaling_enabled', 'N/A')}")

        # Teste Auto-Scaling Aktivierung
        model.enable_auto_scaling()
        print("✅ Auto-Scaling aktiviert")

        # Teste System-Status nach Aktivierung
        status = model.get_system_status()
        print(f"✅ System-Status nach Aktivierung: {status.get('auto_scaling_enabled', 'N/A')}")

        # Teste Performance-Metriken
        metrics = model.get_performance_metrics()
        print(f"✅ Performance-Metriken abgerufen: {len(metrics)} Metriken")

        # Teste manuelle Skalierungsaktion
        model.manual_scaling_action("scale_up", "cpu", 80)
        print("✅ Manuelle Skalierungsaktion ausgeführt")

        # Teste Performance-Optimierung
        model.optimize_performance()
        print("✅ Performance-Optimierung durchgeführt")

        # Teste Auto-Scaling Deaktivierung
        model.disable_auto_scaling()
        print("✅ Auto-Scaling deaktiviert")

        print("🎉 Alle Auto-Scaling Tests erfolgreich!")
        return True

    except Exception as e:
        print(f"❌ Auto-Scaling Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_scaler_direct():
    """Testet den AutoScaler direkt"""
    print("\n🧪 Teste AutoScaler direkt...")

    try:
        # Erstelle AutoScaler Instanz
        scaler = AutoScaler()
        print("✅ AutoScaler erstellt")

        # Teste System-Metriken Erfassung
        metrics = scaler._collect_system_metrics()
        print(f"✅ System-Metriken erfasst: CPU={metrics['cpu_percent']:.1f}%, RAM={metrics['memory_percent']:.1f}%")

        # Teste Performance-Monitoring
        # Sammle einige Metriken zur Historie
        for _ in range(3):
            scaler.performance_monitor.collect_system_metrics()
            time.sleep(0.1)
        print("✅ Performance-Monitoring getestet")

        # Teste Adaptive Optimizer
        decisions = scaler._adaptive_optimizer.analyze_and_optimize()
        if decisions:
            print(f"✅ Adaptive Optimizer Entscheidung: {decisions[0].action} {decisions[0].target}")
        else:
            print("✅ Adaptive Optimizer: Keine Entscheidungen nötig")

        print("🎉 AutoScaler direkte Tests erfolgreich!")
        return True

    except Exception as e:
        print(f"❌ AutoScaler direkter Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hauptfunktion für Tests"""
    print("🚀 Auto-Scaling System Test Suite")
    print("=" * 50)

    # Test 1: Auto-Scaling Integration
    test1_passed = test_auto_scaling_integration()

    # Test 2: AutoScaler direkt
    test2_passed = test_auto_scaler_direct()

    # Zusammenfassung
    print("\n📊 Test-Zusammenfassung:")
    print(f"Auto-Scaling Integration: {'✅' if test1_passed else '❌'}")
    print(f"AutoScaler direkt: {'✅' if test2_passed else '❌'}")

    if test1_passed and test2_passed:
        print("\n🎉 Alle Tests erfolgreich! Auto-Scaling System ist bereit.")
        return 0
    else:
        print("\n❌ Einige Tests fehlgeschlagen. Überprüfen Sie die Implementierung.")
        return 1

if __name__ == "__main__":
    sys.exit(main())