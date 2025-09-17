#!/usr/bin/env python3
"""
Test-Suite für Dynamic Batching System
Testet RTX 2070 optimierte Request-Batching Funktionalität
"""

import asyncio
import logging
import time
from typing import List

logger = logging.getLogger(__name__)


async def test_batch_config():
    """Testet Batch-Konfiguration"""
    try:
        from core.dynamic_batching import DynamicBatchProcessor, BatchConfig

        logger.info("⚙️ Teste Batch-Konfiguration...")

        # Teste RTX 2070 spezifische Konfiguration
        processor = DynamicBatchProcessor(gpu_memory_gb=8.0)

        assert processor.config.max_batch_size == 8, f"RTX 2070 sollte max_batch_size=8 haben, bekam {processor.config.max_batch_size}"
        assert processor.config.max_wait_time == 0.05, f"RTX 2070 sollte max_wait_time=0.05 haben, bekam {processor.config.max_wait_time}"
        assert processor.config.gpu_memory_threshold == 0.85, f"RTX 2070 sollte gpu_memory_threshold=0.85 haben, bekam {processor.config.gpu_memory_threshold}"

        # Teste kleinere GPU
        processor_small = DynamicBatchProcessor(gpu_memory_gb=4.0)
        assert processor_small.config.max_batch_size == 4, f"Kleine GPU sollte max_batch_size=4 haben, bekam {processor_small.config.max_batch_size}"

        logger.info("✅ Batch-Konfiguration funktioniert")

        return True

    except Exception as e:
        logger.error(f"❌ Batch-Konfiguration Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_request_batching():
    """Testet Request-Batching Funktionalität"""
    try:
        from core.dynamic_batching import DynamicBatchProcessor, BatchRequest

        logger.info("📦 Teste Request-Batching...")

        processor = DynamicBatchProcessor(gpu_memory_gb=8.0)

        # Sammle Ergebnisse
        results = []

        async def collect_result(result):
            results.append(result)

        # Submitte mehrere Requests
        requests = []
        for i in range(5):
            req = BatchRequest(
                id=f"req_{i}",
                data=f"data_{i}",
                callback=collect_result
            )
            batch_id = await processor.submit_request(req)
            requests.append(req)

        # Warte auf Verarbeitung
        await asyncio.sleep(0.2)

        # Debug: Prüfe aktuellen Status
        metrics = processor.get_batch_metrics()
        logger.info(f"Debug - Pending: {metrics['pending_requests']}, Processing: {metrics['processing_batches']}, Results: {len(results)}")

        # Prüfe Ergebnisse
        print(f"Bekam {len(results)} Ergebnisse: {results}")
        assert len(results) == 5, f"Erwartet 5 Ergebnisse, bekam {len(results)}"

        # Prüfe Metriken
        metrics = processor.get_batch_metrics()
        assert metrics["total_requests"] == 5, f"Erwartet 5 total_requests, bekam {metrics['total_requests']}"
        assert metrics["total_batches"] >= 1, f"Erwartet mindestens 1 Batch, bekam {metrics['total_batches']}"

        logger.info("✅ Request-Batching funktioniert")

        return True

    except Exception as e:
        logger.error(f"❌ Request-Batching Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_performance():
    """Testet Batch-Performance und Metriken"""
    try:
        from core.dynamic_batching import DynamicBatchProcessor, BatchRequest

        logger.info("📊 Teste Batch-Performance...")

        processor = DynamicBatchProcessor(gpu_memory_gb=8.0)

        # Sammle viele Requests
        num_requests = 20
        results = []

        async def collect_result(result):
            results.append(result)

        start_time = time.time()

        # Submitte Requests in schneller Folge
        for i in range(num_requests):
            req = BatchRequest(
                id=f"perf_req_{i}",
                data=f"perf_data_{i}",
                callback=collect_result
            )
            await processor.submit_request(req)

        # Warte auf vollständige Verarbeitung
        await asyncio.sleep(0.5)

        total_time = time.time() - start_time

        # Prüfe, dass alle Requests verarbeitet wurden
        assert len(results) == num_requests, f"Erwartet {num_requests} Ergebnisse, bekam {len(results)}"

        # Prüfe Performance-Metriken
        metrics = processor.get_batch_metrics()
        assert metrics["total_requests"] == num_requests
        assert metrics["avg_batch_size"] > 0, "Durchschnittliche Batch-Größe sollte > 0 sein"
        assert metrics["throughput_requests_per_sec"] > 0, "Throughput sollte > 0 sein"

        logger.info(f"✅ Batch-Performance: {metrics['throughput_requests_per_sec']:.1f} req/s, avg batch size: {metrics['avg_batch_size']:.1f}")

        return True

    except Exception as e:
        logger.error(f"❌ Batch-Performance Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_adaptive_batching():
    """Testet adaptives Batching basierend auf GPU-Speicher"""
    try:
        from core.dynamic_batching import DynamicBatchProcessor

        logger.info("🎛️ Teste adaptives Batching...")

        # Teste verschiedene GPU-Größen
        gpu_configs = [
            (4.0, 4),   # Kleine GPU
            (8.0, 8),   # RTX 2070
            (12.0, 16), # RTX 3080
            (24.0, 16), # Große GPU
        ]

        for gpu_gb, expected_batch_size in gpu_configs:
            processor = DynamicBatchProcessor(gpu_memory_gb=gpu_gb)
            assert processor.config.max_batch_size == expected_batch_size, \
                f"GPU {gpu_gb}GB sollte max_batch_size={expected_batch_size} haben, bekam {processor.config.max_batch_size}"

            logger.info(f"✅ GPU {gpu_gb}GB: max_batch_size={processor.config.max_batch_size}")

        logger.info("✅ Adaptives Batching funktioniert")

        return True

    except Exception as e:
        logger.error(f"❌ Adaptives Batching Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Hauptfunktion für Dynamic Batching Tests"""
    print("🎯 Dynamic Batching Test Suite")
    print("=" * 50)

    tests = [
        ("Batch Config", test_batch_config),
        ("Request Batching", test_request_batching),
        ("Batch Performance", test_batch_performance),
        ("Adaptive Batching", test_adaptive_batching),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' fehlgeschlagen: {e}")
            results.append((test_name, False))

    # Zusammenfassung
    print("\n" + "="*60)
    print("📋 TEST-ZUSAMMENFASSUNG")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n📊 Ergebnis: {passed}/{total} Tests erfolgreich")

    if passed == total:
        print("🎉 Alle Dynamic Batching Tests erfolgreich!")
        print("✅ RTX 2070 optimierte Batch-Verarbeitung aktiv")
        print("✅ Adaptive GPU-Auslastung funktioniert")
        print("✅ Performance-Monitoring bereit")
    elif passed >= total * 0.8:
        print("👍 Meisten Tests erfolgreich. Dynamic Batching teilweise aktiv.")
    else:
        print("⚠️ Einige Tests fehlgeschlagen. Fallback auf sequentielle Verarbeitung.")

    print("\n💡 Dynamic Batching Features:")
    print("   • RTX 2070 spezifische Batch-Größenoptimierung")
    print("   • Adaptive Batch-Verarbeitung basierend auf GPU-Speicher")
    print("   • Performance-Monitoring und Metriken")
    print("   • Asynchrone Request-Verarbeitung")
    print("   • Memory-optimierte GPU-Nutzung")


if __name__ == "__main__":
    asyncio.run(main())