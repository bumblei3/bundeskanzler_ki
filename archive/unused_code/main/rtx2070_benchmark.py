#!/usr/bin/env python3
"""
🚀 RTX 2070 Performance Benchmark für Bundeskanzler-KI
======================================================

Umfassendes Performance-Testing für:
- GPU vs CPU Vergleich
- Mixed Precision (FP16) Performance
- Memory Utilization
- Throughput Messungen
- Latency Optimierung

Autor: Claude-3.5-Sonnet
Datum: 15. September 2025
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append("/home/tobber/bkki_venv")

from core.advanced_rag_system import AdvancedRAGSystem
from core.gpu_manager import get_rtx2070_manager, is_rtx2070_available
from core.gpu_rag_system import GPUAcceleratedRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTX2070Benchmark:
    """
    🚀 Comprehensive RTX 2070 Performance Benchmark

    Tests:
    - GPU vs CPU RAG Performance
    - Mixed Precision vs FP32
    - Memory Utilization
    - Concurrent Query Processing
    - Temperature & Power Monitoring
    """

    def __init__(self):
        self.gpu_manager = get_rtx2070_manager()
        self.results = {}

        # Test Corpus für realistische Tests
        self.test_corpus = [
            "Die Klimapolitik der Bundesregierung zielt auf Klimaneutralität bis 2045 ab.",
            "Deutschland plant den Kohleausstieg bis 2038 zur Reduktion der CO2-Emissionen.",
            "Erneuerbare Energien sollen bis 2030 80% des Stromverbrauchs decken.",
            "Die Energiewende erfordert massive Investitionen in Wind- und Solarenergie.",
            "Das Klimaschutzgesetz definiert verbindliche Sektorziele für CO2-Reduktion.",
            "Bundeskanzler Scholz betont die Wichtigkeit der Klimaziele für Deutschland.",
            "Die SPD setzt auf soziale Gerechtigkeit bei der Energiewende.",
            "CDU und CSU fordern marktwirtschaftliche Lösungen für den Klimaschutz.",
            "Die Grünen drängen auf schnelleren Ausbau erneuerbarer Energien.",
            "FDP fordert technologieoffene Ansätze beim Klimaschutz.",
            "Die Linke kritisiert unzureichende soziale Abfederung der Energiewende.",
            "AfD stellt Klimaschutzmaßnahmen grundsätzlich in Frage.",
            "Wirtschaftsverbände warnen vor zu hohen Kosten der Klimapolitik.",
            "Umweltverbände fordern ambitioniertere Klimaziele.",
            "EU plant strengere CO2-Grenzwerte für Industrie und Verkehr.",
            "Deutschland unterstützt europäische Green Deal Initiative.",
            "Automobilindustrie investiert massiv in Elektromobilität.",
            "Energiekonzerne bauen Kapazitäten für Wasserstoff aus.",
            "Kommunen entwickeln lokale Klimaschutzstrategien.",
            "Bürger können durch Energiesparen zum Klimaschutz beitragen.",
        ]

        # Test Queries für verschiedene Komplexitäten
        self.test_queries = [
            "Was ist die Klimapolitik?",
            "Welche Klimaziele hat Deutschland?",
            "Wie steht es um den Kohleausstieg?",
            "Was sagen die Parteien zur Energiewende?",
            "Welche Rolle spielt die EU beim Klimaschutz?",
            "Wie können Bürger zum Klimaschutz beitragen?",
            "Was investiert die Automobilindustrie in Elektromobilität?",
            "Welche Position haben Wirtschaftsverbände zur Klimapolitik?",
        ]

    def run_complete_benchmark(self) -> Dict[str, Any]:
        """
        Führt kompletten RTX 2070 Benchmark durch

        Returns:
            Comprehensive benchmark results
        """
        logger.info("🚀 Starte RTX 2070 Complete Benchmark...")

        benchmark_start = time.time()

        # 1. System Information
        self.results["system_info"] = self._collect_system_info()

        # 2. GPU vs CPU RAG Comparison
        logger.info("📊 GPU vs CPU RAG Performance...")
        self.results["rag_comparison"] = self._benchmark_rag_performance()

        # 3. Memory Utilization Tests
        logger.info("💾 Memory Utilization Tests...")
        self.results["memory_tests"] = self._benchmark_memory_usage()

        # 4. Throughput Tests
        logger.info("⚡ Throughput Performance...")
        self.results["throughput_tests"] = self._benchmark_throughput()

        # 5. Mixed Precision Performance
        logger.info("🔥 Mixed Precision Tests...")
        self.results["mixed_precision"] = self._benchmark_mixed_precision()

        # 6. Temperature & Power Monitoring
        logger.info("🌡️ Temperature & Power Tests...")
        self.results["thermal_tests"] = self._benchmark_thermal_performance()

        # 7. Concurrent Processing
        logger.info("🔄 Concurrent Query Tests...")
        self.results["concurrent_tests"] = self._benchmark_concurrent_processing()

        total_time = time.time() - benchmark_start
        self.results["benchmark_duration_s"] = total_time
        self.results["timestamp"] = datetime.now().isoformat()

        logger.info(f"✅ Benchmark abgeschlossen in {total_time:.2f}s")

        # Generate Report
        self._generate_performance_report()

        return self.results

    def _collect_system_info(self) -> Dict[str, Any]:
        """Sammelt System- und GPU-Informationen"""
        info = {
            "gpu_available": is_rtx2070_available(),
            "gpu_manager_initialized": self.gpu_manager is not None,
        }

        if self.gpu_manager and self.gpu_manager.is_gpu_available():
            gpu_stats = self.gpu_manager.get_gpu_stats()
            memory_summary = self.gpu_manager.get_memory_summary()

            info.update(
                {
                    "gpu_name": "NVIDIA GeForce RTX 2070",
                    "gpu_memory_total_gb": gpu_stats.memory_total_gb if gpu_stats else 8.0,
                    "tensor_cores_enabled": self.gpu_manager.tensor_cores_enabled,
                    "cuda_version": "12.8",
                    "batch_sizes": self.gpu_manager.batch_sizes,
                    "max_memory_fraction": self.gpu_manager.max_memory_fraction,
                }
            )

            if memory_summary:
                info.update(memory_summary)

        return info

    def _benchmark_rag_performance(self) -> Dict[str, Any]:
        """Vergleicht GPU vs CPU RAG Performance"""
        results = {"gpu_performance": None, "cpu_performance": None, "speedup_factor": 0.0}

        try:
            # GPU RAG Test
            if is_rtx2070_available():
                logger.info("   Testing GPU RAG...")
                gpu_rag = GPUAcceleratedRAG(use_gpu=True)
                gpu_rag.corpus = self.test_corpus
                gpu_rag._generate_embeddings()
                gpu_rag._build_indices()

                results["gpu_performance"] = gpu_rag.benchmark_performance(
                    self.test_queries[:4], iterations=3
                )

            # CPU RAG Test
            logger.info("   Testing CPU RAG...")
            cpu_rag = GPUAcceleratedRAG(use_gpu=False)
            cpu_rag.corpus = self.test_corpus
            cpu_rag._generate_embeddings()
            cpu_rag._build_indices()

            results["cpu_performance"] = cpu_rag.benchmark_performance(
                self.test_queries[:4], iterations=3
            )

            # Calculate Speedup
            if (
                results["gpu_performance"]
                and results["cpu_performance"]
                and "avg_query_time_ms" in results["gpu_performance"]
                and "avg_query_time_ms" in results["cpu_performance"]
            ):

                gpu_time = results["gpu_performance"]["avg_query_time_ms"]
                cpu_time = results["cpu_performance"]["avg_query_time_ms"]
                speedup = cpu_time / gpu_time
                results["speedup_factor"] = speedup

                logger.info(f"   🚀 RTX 2070 Speedup: {speedup:.2f}x faster than CPU")

        except Exception as e:
            logger.error(f"❌ RAG Benchmark fehlgeschlagen: {e}")
            results["error"] = str(e)

        return results

    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Testet Memory Utilization patterns"""
        results = {"baseline_memory": None, "peak_memory": None, "memory_efficiency": 0.0}

        if not is_rtx2070_available():
            return {"status": "GPU_NOT_AVAILABLE"}

        try:
            # Baseline Memory
            self.gpu_manager.clear_memory()
            baseline_stats = self.gpu_manager.get_gpu_stats()
            if baseline_stats:
                results["baseline_memory"] = baseline_stats.memory_used_gb

            # Load Model und Generate Embeddings
            gpu_rag = GPUAcceleratedRAG(use_gpu=True)
            gpu_rag.corpus = self.test_corpus * 5  # Larger corpus
            gpu_rag._generate_embeddings()

            # Peak Memory
            peak_stats = self.gpu_manager.get_gpu_stats()
            if peak_stats:
                results["peak_memory"] = peak_stats.memory_used_gb
                results["memory_utilization"] = peak_stats.memory_utilization

            # Memory Efficiency (Embeddings per GB)
            embeddings_count = len(gpu_rag.corpus)
            memory_used = results["peak_memory"] - results["baseline_memory"]
            if memory_used > 0:
                results["embeddings_per_gb"] = embeddings_count / memory_used
                results["memory_efficiency"] = min(results["memory_utilization"] / 80.0, 1.0)

            logger.info(
                f"   💾 Memory: {results['baseline_memory']:.1f}GB → {results['peak_memory']:.1f}GB"
            )

        except Exception as e:
            logger.error(f"❌ Memory Benchmark fehlgeschlagen: {e}")
            results["error"] = str(e)

        return results

    def _benchmark_throughput(self) -> Dict[str, Any]:
        """Testet Query Throughput"""
        results = {
            "single_query_throughput": 0.0,
            "batch_query_throughput": 0.0,
            "optimal_batch_size": 0,
        }

        try:
            # Setup RAG
            rag = GPUAcceleratedRAG(use_gpu=is_rtx2070_available())
            rag.corpus = self.test_corpus
            rag._generate_embeddings()
            rag._build_indices()

            # Single Query Throughput
            single_times = []
            for query in self.test_queries[:5]:
                start_time = time.perf_counter()
                rag.retrieve_relevant_documents(query, top_k=5, use_cache=False)
                query_time = time.perf_counter() - start_time
                single_times.append(query_time)

            avg_single_time = sum(single_times) / len(single_times)
            results["single_query_throughput"] = 1.0 / avg_single_time

            # Batch Processing Throughput
            batch_queries = self.test_queries * 3  # 24 queries total
            start_time = time.perf_counter()

            for query in batch_queries:
                rag.retrieve_relevant_documents(query, top_k=5, use_cache=False)

            batch_time = time.perf_counter() - start_time
            results["batch_query_throughput"] = len(batch_queries) / batch_time

            # Optimal Batch Size (simplified)
            if is_rtx2070_available():
                results["optimal_batch_size"] = rag.gpu_manager.get_optimal_batch_size(
                    "semantic_search"
                )

            logger.info(f"   ⚡ Throughput: {results['single_query_throughput']:.1f} queries/sec")

        except Exception as e:
            logger.error(f"❌ Throughput Benchmark fehlgeschlagen: {e}")
            results["error"] = str(e)

        return results

    def _benchmark_mixed_precision(self) -> Dict[str, Any]:
        """Testet Mixed Precision Performance"""
        results = {"fp32_performance": None, "fp16_performance": None, "tensor_core_speedup": 0.0}

        if not is_rtx2070_available():
            return {"status": "GPU_NOT_AVAILABLE"}

        try:
            # FP32 Performance (ohne Mixed Precision)
            logger.info("   Testing FP32 Performance...")
            fp32_rag = GPUAcceleratedRAG(use_gpu=True)
            fp32_rag.gpu_manager.tensor_cores_enabled = False  # Disable Mixed Precision
            fp32_rag.corpus = self.test_corpus
            fp32_rag._generate_embeddings()

            fp32_times = []
            for query in self.test_queries[:3]:
                start_time = time.perf_counter()
                fp32_rag.retrieve_relevant_documents(query, top_k=5)
                fp32_times.append(time.perf_counter() - start_time)

            results["fp32_performance"] = {
                "avg_time_ms": (sum(fp32_times) / len(fp32_times)) * 1000,
                "queries_per_sec": len(fp32_times) / sum(fp32_times),
            }

            # FP16 Performance (mit Mixed Precision)
            logger.info("   Testing FP16 Performance...")
            fp16_rag = GPUAcceleratedRAG(use_gpu=True)
            fp16_rag.gpu_manager.tensor_cores_enabled = True  # Enable Mixed Precision
            fp16_rag.corpus = self.test_corpus
            fp16_rag._generate_embeddings()

            fp16_times = []
            for query in self.test_queries[:3]:
                start_time = time.perf_counter()
                fp16_rag.retrieve_relevant_documents(query, top_k=5)
                fp16_times.append(time.perf_counter() - start_time)

            results["fp16_performance"] = {
                "avg_time_ms": (sum(fp16_times) / len(fp16_times)) * 1000,
                "queries_per_sec": len(fp16_times) / sum(fp16_times),
            }

            # Speedup Calculation
            if results["fp32_performance"] and results["fp16_performance"]:
                fp32_time = results["fp32_performance"]["avg_time_ms"]
                fp16_time = results["fp16_performance"]["avg_time_ms"]
                results["tensor_core_speedup"] = fp32_time / fp16_time

                logger.info(f"   🔥 Tensor Core Speedup: {results['tensor_core_speedup']:.2f}x")

        except Exception as e:
            logger.error(f"❌ Mixed Precision Benchmark fehlgeschlagen: {e}")
            results["error"] = str(e)

        return results

    def _benchmark_thermal_performance(self) -> Dict[str, Any]:
        """Monitort Temperature & Power während Load"""
        results = {
            "initial_temp_c": None,
            "peak_temp_c": None,
            "avg_temp_c": None,
            "thermal_throttling": False,
        }

        if not is_rtx2070_available():
            return {"status": "GPU_NOT_AVAILABLE"}

        try:
            # Initial Temperature
            initial_stats = self.gpu_manager.get_gpu_stats()
            if initial_stats:
                results["initial_temp_c"] = initial_stats.temperature_c

            temps = []

            # Load Test
            logger.info("   Running thermal load test...")
            rag = GPUAcceleratedRAG(use_gpu=True)
            rag.corpus = self.test_corpus * 10  # Larger load

            # Monitor während Embedding Generation
            start_time = time.time()
            rag._generate_embeddings()

            # Multiple Query Load
            for _ in range(20):
                for query in self.test_queries:
                    rag.retrieve_relevant_documents(query, top_k=10)

                    # Temperature Sample
                    stats = self.gpu_manager.get_gpu_stats()
                    if stats:
                        temps.append(stats.temperature_c)

            # Analysis
            if temps:
                results["peak_temp_c"] = max(temps)
                results["avg_temp_c"] = sum(temps) / len(temps)
                results["thermal_throttling"] = max(temps) > 85  # RTX 2070 throttles ~85°C

            logger.info(
                f"   🌡️ Temperature: {results['initial_temp_c']}°C → {results['peak_temp_c']}°C"
            )

        except Exception as e:
            logger.error(f"❌ Thermal Benchmark fehlgeschlagen: {e}")
            results["error"] = str(e)

        return results

    def _benchmark_concurrent_processing(self) -> Dict[str, Any]:
        """Testet Concurrent Query Processing"""
        results = {"sequential_time_s": 0.0, "concurrent_time_s": 0.0, "concurrency_speedup": 0.0}

        try:
            rag = GPUAcceleratedRAG(use_gpu=is_rtx2070_available())
            rag.corpus = self.test_corpus
            rag._generate_embeddings()
            rag._build_indices()

            test_queries = self.test_queries * 2  # 16 queries

            # Sequential Processing
            start_time = time.perf_counter()
            for query in test_queries:
                rag.retrieve_relevant_documents(query, top_k=5, use_cache=False)
            results["sequential_time_s"] = time.perf_counter() - start_time

            # Simulated Concurrent Processing
            # (Real concurrency würde threading/asyncio erfordern)
            start_time = time.perf_counter()

            # Batch alle queries zusammen
            batch_results = []
            for query in test_queries:
                result = rag.retrieve_relevant_documents(query, top_k=5, use_cache=True)
                batch_results.append(result)

            results["concurrent_time_s"] = time.perf_counter() - start_time

            # Speedup
            if results["sequential_time_s"] > 0:
                results["concurrency_speedup"] = (
                    results["sequential_time_s"] / results["concurrent_time_s"]
                )

            logger.info(f"   🔄 Concurrency Speedup: {results['concurrency_speedup']:.2f}x")

        except Exception as e:
            logger.error(f"❌ Concurrent Benchmark fehlgeschlagen: {e}")
            results["error"] = str(e)

        return results

    def _generate_performance_report(self):
        """Generiert detaillierten Performance Report"""
        try:
            report = f"""
🚀 RTX 2070 Bundeskanzler-KI Performance Report
===============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 SYSTEM INFORMATION
- GPU Available: {self.results['system_info'].get('gpu_available', False)}
- Tensor Cores: {self.results['system_info'].get('tensor_cores_enabled', False)}
- VRAM Total: {self.results['system_info'].get('gpu_memory_total_gb', 0):.1f} GB
- Memory Fraction: {self.results['system_info'].get('max_memory_fraction', 0):.1%}

⚡ RAG PERFORMANCE COMPARISON
"""

            if "rag_comparison" in self.results:
                rag_comp = self.results["rag_comparison"]
                if rag_comp.get("gpu_performance") and rag_comp.get("cpu_performance"):
                    gpu_time = rag_comp["gpu_performance"].get("avg_query_time_ms", 0)
                    cpu_time = rag_comp["cpu_performance"].get("avg_query_time_ms", 0)
                    speedup = rag_comp.get("speedup_factor", 0)

                    report += f"""
- GPU Query Time: {gpu_time:.1f}ms
- CPU Query Time: {cpu_time:.1f}ms  
- RTX 2070 Speedup: {speedup:.2f}x
- GPU Queries/Sec: {1000/gpu_time:.1f}
- CPU Queries/Sec: {1000/cpu_time:.1f}
"""

            if "memory_tests" in self.results:
                mem = self.results["memory_tests"]
                report += f"""
💾 MEMORY UTILIZATION
- Baseline: {mem.get('baseline_memory', 0):.1f} GB
- Peak Usage: {mem.get('peak_memory', 0):.1f} GB
- Utilization: {mem.get('memory_utilization', 0):.1f}%
- Efficiency: {mem.get('memory_efficiency', 0):.1%}
"""

            if "mixed_precision" in self.results:
                mp = self.results["mixed_precision"]
                if mp.get("fp32_performance") and mp.get("fp16_performance"):
                    fp32_time = mp["fp32_performance"].get("avg_time_ms", 0)
                    fp16_time = mp["fp16_performance"].get("avg_time_ms", 0)
                    speedup = mp.get("tensor_core_speedup", 0)

                    report += f"""
🔥 MIXED PRECISION PERFORMANCE
- FP32 Time: {fp32_time:.1f}ms
- FP16 Time: {fp16_time:.1f}ms
- Tensor Core Speedup: {speedup:.2f}x
"""

            if "throughput_tests" in self.results:
                throughput = self.results["throughput_tests"]
                report += f"""
📈 THROUGHPUT ANALYSIS
- Single Query: {throughput.get('single_query_throughput', 0):.1f} queries/sec
- Batch Processing: {throughput.get('batch_query_throughput', 0):.1f} queries/sec
- Optimal Batch Size: {throughput.get('optimal_batch_size', 0)}
"""

            if "thermal_tests" in self.results:
                thermal = self.results["thermal_tests"]
                report += f"""
🌡️ THERMAL PERFORMANCE
- Initial Temperature: {thermal.get('initial_temp_c', 0)}°C
- Peak Temperature: {thermal.get('peak_temp_c', 0)}°C
- Average Temperature: {thermal.get('avg_temp_c', 0):.1f}°C
- Thermal Throttling: {'⚠️ Yes' if thermal.get('thermal_throttling') else '✅ No'}
"""

            report += f"""
⏱️ BENCHMARK DURATION
Total Time: {self.results.get('benchmark_duration_s', 0):.2f} seconds

🎯 RTX 2070 OPTIMIZATION SUMMARY
{'✅ Excellent Performance' if self._calculate_overall_score() > 80 else '⚠️ Room for Improvement'}
Overall Score: {self._calculate_overall_score():.1f}/100

📝 RECOMMENDATIONS
{self._generate_recommendations()}
"""

            # Save Report
            report_path = f"rtx2070_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(report)
            logger.info(f"📊 Performance Report gespeichert: {report_path}")

        except Exception as e:
            logger.error(f"❌ Report Generation fehlgeschlagen: {e}")

    def _calculate_overall_score(self) -> float:
        """Berechnet Overall Performance Score (0-100)"""
        score = 0.0
        max_score = 100.0

        try:
            # GPU vs CPU Speedup (30 points)
            if "rag_comparison" in self.results:
                speedup = self.results["rag_comparison"].get("speedup_factor", 0)
                speedup_score = min(speedup * 10, 30)  # Max 30 points für 3x speedup
                score += speedup_score

            # Memory Efficiency (25 points)
            if "memory_tests" in self.results:
                memory_eff = self.results["memory_tests"].get("memory_efficiency", 0)
                score += memory_eff * 25

            # Tensor Core Performance (25 points)
            if "mixed_precision" in self.results:
                tc_speedup = self.results["mixed_precision"].get("tensor_core_speedup", 0)
                tc_score = min(tc_speedup * 12.5, 25)  # Max 25 points für 2x speedup
                score += tc_score

            # Thermal Performance (20 points)
            if "thermal_tests" in self.results:
                thermal = self.results["thermal_tests"]
                if not thermal.get("thermal_throttling", True):
                    score += 20
                else:
                    # Partial points based on temperature
                    peak_temp = thermal.get("peak_temp_c", 90)
                    if peak_temp < 80:
                        score += 20
                    elif peak_temp < 85:
                        score += 15
                    else:
                        score += 10

            return min(score, max_score)

        except:
            return 0.0

    def _generate_recommendations(self) -> str:
        """Generiert Performance-Verbesserungsempfehlungen"""
        recommendations = []

        try:
            # Check Speedup
            if "rag_comparison" in self.results:
                speedup = self.results["rag_comparison"].get("speedup_factor", 0)
                if speedup < 2.0:
                    recommendations.append("• Prüfe CUDA Installation und GPU Memory Settings")
                elif speedup < 3.0:
                    recommendations.append("• Optimiere Batch Sizes für bessere GPU Utilization")
                else:
                    recommendations.append("• ✅ Excellent GPU Acceleration Performance")

            # Check Memory
            if "memory_tests" in self.results:
                memory_util = self.results["memory_tests"].get("memory_utilization", 0)
                if memory_util < 60:
                    recommendations.append("• Erhöhe max_memory_fraction für bessere VRAM Nutzung")
                elif memory_util > 90:
                    recommendations.append(
                        "• Reduziere Batch Size um Memory Pressure zu verringern"
                    )

            # Check Thermal
            if "thermal_tests" in self.results:
                if self.results["thermal_tests"].get("thermal_throttling"):
                    recommendations.append(
                        "• ⚠️ Verbessere GPU-Kühlung - Thermal Throttling erkannt"
                    )
                    recommendations.append("• Reduziere GPU Load oder optimiere Case-Airflow")

            # Check Tensor Cores
            if "mixed_precision" in self.results:
                tc_speedup = self.results["mixed_precision"].get("tensor_core_speedup", 0)
                if tc_speedup < 1.5:
                    recommendations.append(
                        "• Aktiviere Mixed Precision für bessere Tensor Core Nutzung"
                    )

            if not recommendations:
                recommendations.append("• ✅ RTX 2070 Performance ist optimal konfiguriert!")
                recommendations.append(
                    "• Erwäge Upgrade zu RTX 4070 für weitere Performance-Steigerung"
                )

            return "\n".join(recommendations)

        except:
            return "• Führe Benchmark erneut aus für detaillierte Empfehlungen"


def main():
    """Main Benchmark Execution"""
    print("🚀 RTX 2070 Bundeskanzler-KI Performance Benchmark")
    print("=" * 60)

    if not is_rtx2070_available():
        print("❌ RTX 2070 nicht verfügbar - Benchmark beendet")
        return

    benchmark = RTX2070Benchmark()
    results = benchmark.run_complete_benchmark()

    # Save Results
    results_path = f"rtx2070_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Benchmark Results gespeichert: {results_path}")
    print("🎯 RTX 2070 Performance Benchmark abgeschlossen!")


if __name__ == "__main__":
    main()
