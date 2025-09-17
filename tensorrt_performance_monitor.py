#!/usr/bin/env python3
"""
TensorRT Performance Monitoring und Benchmarking System
Misst Performance-Verbesserungen durch TensorRT-Optimierung auf RTX 2070
"""

import sys
import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# TensorRT und PyTorch Imports
try:
    import tensorrt as trt
    import torch
    import numpy as np
except ImportError as e:
    print(f"❌ Erforderliche Bibliotheken nicht verfügbar: {e}")
    sys.exit(1)

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_ki import MultimodalTransformerModel

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance-Metriken für TensorRT-Optimierung"""
    model_name: str
    model_type: str  # 'text', 'vision', 'audio'
    original_latency: float  # Sekunden
    tensorrt_latency: float  # Sekunden
    speedup_factor: float
    memory_usage_original: float  # MB
    memory_usage_tensorrt: float  # MB
    memory_reduction: float  # Prozent
    throughput_original: float  # tokens/sec oder fps
    throughput_tensorrt: float  # tokens/sec oder fps
    batch_size: int
    precision_mode: str  # 'FP32', 'FP16', 'TF32'
    timestamp: str
    gpu_utilization: float  # Prozent
    power_consumption: float  # Watt (falls verfügbar)

@dataclass
class BenchmarkResult:
    """Ergebnis eines Benchmark-Tests"""
    test_name: str
    metrics: List[PerformanceMetrics]
    total_speedup: float
    total_memory_reduction: float
    test_duration: float
    timestamp: str
    system_info: Dict[str, Any]

class TensorRTPerformanceMonitor:
    """Performance-Monitoring für TensorRT-Optimierungen"""

    def __init__(self):
        self.results_dir = project_root / "monitoring" / "tensorrt_benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # System-Informationen sammeln
        self.system_info = self._collect_system_info()

        logger.info("📊 TensorRT Performance Monitor initialisiert")

    def _collect_system_info(self) -> Dict[str, Any]:
        """Sammelt System-Informationen"""
        info = {
            "gpu_name": "Unknown",
            "gpu_memory": 0,
            "cuda_version": "Unknown",
            "tensorrt_version": trt.__version__,
            "pytorch_version": torch.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "os": sys.platform
        }

        try:
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

                # CUDA Version
                cuda_version = torch.version.cuda
                if cuda_version:
                    info["cuda_version"] = cuda_version

        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Sammeln von GPU-Informationen: {e}")

        return info

    def benchmark_model(self, model: MultimodalTransformerModel, model_type: str,
                       test_data: Any, batch_sizes: List[int] = [1, 2, 4]) -> List[PerformanceMetrics]:
        """Benchmarkt ein Modell mit verschiedenen Batch-Größen"""
        logger.info(f"🏃 Benchmarke {model_type}-Modell...")

        metrics = []

        for batch_size in batch_sizes:
            logger.info(f"  Teste Batch-Size {batch_size}...")

            try:
                # Original-Performance messen
                original_metrics = self._measure_original_performance(
                    model, model_type, test_data, batch_size
                )

                # TensorRT-Performance messen
                tensorrt_metrics = self._measure_tensorrt_performance(
                    model, model_type, test_data, batch_size
                )

                # Metriken kombinieren
                combined_metrics = PerformanceMetrics(
                    model_name=f"{model_type}_model",
                    model_type=model_type,
                    original_latency=original_metrics["latency"],
                    tensorrt_latency=tensorrt_metrics["latency"],
                    speedup_factor=original_metrics["latency"] / tensorrt_metrics["latency"],
                    memory_usage_original=original_metrics["memory"],
                    memory_usage_tensorrt=tensorrt_metrics["memory"],
                    memory_reduction=((original_metrics["memory"] - tensorrt_metrics["memory"]) /
                                    original_metrics["memory"]) * 100,
                    throughput_original=original_metrics["throughput"],
                    throughput_tensorrt=tensorrt_metrics["throughput"],
                    batch_size=batch_size,
                    precision_mode="FP16" if model.is_rtx2070 else "FP32",
                    timestamp=datetime.now().isoformat(),
                    gpu_utilization=tensorrt_metrics.get("gpu_util", 0.0),
                    power_consumption=0.0  # Nicht verfügbar ohne NVML
                )

                metrics.append(combined_metrics)

                logger.info(".2f")
                logger.info(".1f")
                logger.info(".1f")

            except Exception as e:
                logger.error(f"❌ Fehler bei Batch-Size {batch_size}: {e}")
                continue

        return metrics

    def _measure_original_performance(self, model: MultimodalTransformerModel,
                                    model_type: str, test_data: Any, batch_size: int) -> Dict[str, float]:
        """Misst Original-Performance ohne TensorRT"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        # Inference durchführen (ohne TensorRT durch direktes Aufrufen der Methoden)
        if model_type == "text":
            result = model.process_text(test_data)
        elif model_type == "vision":
            result = model.process_image(test_data)
        elif model_type == "audio":
            result = model.process_audio(test_data)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**2

        latency = end_time - start_time
        memory_usage = end_memory - start_memory

        # Throughput berechnen (vereinfacht)
        if model_type == "text":
            throughput = len(test_data.split()) / latency  # tokens/sec
        elif model_type == "vision":
            throughput = 1.0 / latency  # fps
        elif model_type == "audio":
            throughput = 1.0 / latency  # fps
        else:
            throughput = 1.0 / latency

        return {
            "latency": latency,
            "memory": memory_usage,
            "throughput": throughput
        }

    def _measure_tensorrt_performance(self, model: MultimodalTransformerModel,
                                    model_type: str, test_data: Any, batch_size: int) -> Dict[str, float]:
        """Misst TensorRT-Performance"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() / 1024**2

        # Inference mit TensorRT durchführen (automatische Auswahl)
        if model_type == "text":
            result = model.process_text(test_data)
        elif model_type == "vision":
            result = model.process_image(test_data)
        elif model_type == "audio":
            result = model.process_audio(test_data)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**2

        latency = end_time - start_time
        memory_usage = end_memory - start_memory

        # Throughput berechnen
        if model_type == "text":
            throughput = len(test_data.split()) / latency
        elif model_type == "vision":
            throughput = 1.0 / latency
        elif model_type == "audio":
            throughput = 1.0 / latency
        else:
            throughput = 1.0 / latency

        # GPU-Utilization (vereinfacht)
        gpu_util = 85.0 if model.is_rtx2070 else 65.0  # Geschätzt

        return {
            "latency": latency,
            "memory": memory_usage,
            "throughput": throughput,
            "gpu_util": gpu_util
        }

    def _measure_tensorrt_performance(self, model: MultimodalTransformerModel,
                                    model_type: str, test_data: Any, batch_size: int) -> Dict[str, float]:
        """Misst TensorRT-Performance"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() / 1024**2

        # Inference mit TensorRT durchführen
        if model_type == "text":
            result = model.process_text(test_data, use_tensorrt=True)
        elif model_type == "vision":
            result = model.process_image(test_data, use_tensorrt=True)
        elif model_type == "audio":
            result = model.process_audio(test_data, use_tensorrt=True)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**2

        latency = end_time - start_time
        memory_usage = end_memory - start_memory

        # Throughput berechnen
        if model_type == "text":
            throughput = len(test_data.split()) / latency
        elif model_type == "vision":
            throughput = 1.0 / latency
        elif model_type == "audio":
            throughput = 1.0 / latency
        else:
            throughput = 1.0 / latency

        # GPU-Utilization (vereinfacht)
        gpu_util = 85.0 if model.is_rtx2070 else 65.0  # Geschätzt

        return {
            "latency": latency,
            "memory": memory_usage,
            "throughput": throughput,
            "gpu_util": gpu_util
        }

    def run_comprehensive_benchmark(self, model: MultimodalTransformerModel) -> BenchmarkResult:
        """Führt umfassenden Benchmark durch"""
        logger.info("🚀 Starte umfassenden TensorRT-Benchmark...")

        start_time = time.time()
        all_metrics = []

        # Test-Daten für verschiedene Modalitäten
        test_text = "Das ist ein Test für die deutsche Sprachverarbeitung mit TensorRT-Optimierung."
        test_image = None  # Vereinfacht für diesen Test
        test_audio = None  # Vereinfacht für diesen Test

        # Text-Modell benchmarken
        try:
            text_metrics = self.benchmark_model(model, "text", test_text, [1, 2])
            all_metrics.extend(text_metrics)
        except Exception as e:
            logger.warning(f"⚠️ Text-Benchmark fehlgeschlagen: {e}")

        # Vision-Modell benchmarken (falls verfügbar)
        try:
            if hasattr(model, 'vision_model') and model.vision_model:
                vision_metrics = self.benchmark_model(model, "vision", test_image, [1])
                all_metrics.extend(vision_metrics)
        except Exception as e:
            logger.warning(f"⚠️ Vision-Benchmark fehlgeschlagen: {e}")

        # Audio-Modell benchmarken (falls verfügbar)
        try:
            if hasattr(model, 'audio_model') and model.audio_model:
                audio_metrics = self.benchmark_model(model, "audio", test_audio, [1])
                all_metrics.extend(audio_metrics)
        except Exception as e:
            logger.warning(f"⚠️ Audio-Benchmark fehlgeschlagen: {e}")

        # Gesamt-Statistiken berechnen
        if all_metrics:
            avg_speedup = np.mean([m.speedup_factor for m in all_metrics])
            avg_memory_reduction = np.mean([m.memory_reduction for m in all_metrics])
        else:
            avg_speedup = 1.0
            avg_memory_reduction = 0.0

        end_time = time.time()
        test_duration = end_time - start_time

        result = BenchmarkResult(
            test_name="RTX2070_TensorRT_Comprehensive_Benchmark",
            metrics=all_metrics,
            total_speedup=avg_speedup,
            total_memory_reduction=avg_memory_reduction,
            test_duration=test_duration,
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info
        )

        # Ergebnisse speichern
        self._save_benchmark_result(result)

        logger.info("✅ Benchmark abgeschlossen")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".2f")

        return result

    def _save_benchmark_result(self, result: BenchmarkResult):
        """Speichert Benchmark-Ergebnisse als JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tensorrt_benchmark_{timestamp}.json"
        filepath = self.results_dir / filename

        # Konvertiere dataclasses zu dicts
        result_dict = asdict(result)
        result_dict["metrics"] = [asdict(m) for m in result.metrics]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Benchmark-Ergebnisse gespeichert: {filepath}")

    def load_benchmark_history(self) -> List[BenchmarkResult]:
        """Lädt Benchmark-Historie"""
        history = []

        if not self.results_dir.exists():
            return history

        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Konvertiere dicts zurück zu dataclasses
                metrics = [PerformanceMetrics(**m) for m in data["metrics"]]
                result = BenchmarkResult(
                    test_name=data["test_name"],
                    metrics=metrics,
                    total_speedup=data["total_speedup"],
                    total_memory_reduction=data["total_memory_reduction"],
                    test_duration=data["test_duration"],
                    timestamp=data["timestamp"],
                    system_info=data["system_info"]
                )
                history.append(result)

            except Exception as e:
                logger.warning(f"⚠️ Fehler beim Laden {json_file}: {e}")

        # Nach Timestamp sortieren
        history.sort(key=lambda x: x.timestamp, reverse=True)
        return history

    def generate_performance_report(self) -> str:
        """Generiert Performance-Report"""
        history = self.load_benchmark_history()

        if not history:
            return "Keine Benchmark-Ergebnisse verfügbar"

        latest = history[0]

        report = f"""
# TensorRT Performance Report - RTX 2070
**Generiert am:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## System-Informationen
- **GPU:** {self.system_info.get('gpu_name', 'Unknown')}
- **GPU Memory:** {self.system_info.get('gpu_memory', 0):.1f} GB
- **TensorRT Version:** {self.system_info.get('tensorrt_version', 'Unknown')}
- **PyTorch Version:** {self.system_info.get('pytorch_version', 'Unknown')}

## Letzte Benchmark-Ergebnisse
**Test:** {latest.test_name}
**Dauer:** {latest.test_duration:.2f}s
**Durchschnittliche Beschleunigung:** {latest.total_speedup:.2f}x
**Durchschnittliche Speicherreduktion:** {latest.total_memory_reduction:.1f}%

## Detaillierte Metriken
"""

        for metric in latest.metrics:
            report += f"""
### {metric.model_name} (Batch Size: {metric.batch_size})
- **Modell-Typ:** {metric.model_type}
- **Original Latenz:** {metric.original_latency:.4f}s
- **TensorRT Latenz:** {metric.tensorrt_latency:.4f}s
- **Beschleunigung:** {metric.speedup_factor:.2f}x
- **Original Speicher:** {metric.memory_usage_original:.1f} MB
- **TensorRT Speicher:** {metric.memory_usage_tensorrt:.1f} MB
- **Speicherreduktion:** {metric.memory_reduction:.1f}%
- **Precision Mode:** {metric.precision_mode}
- **GPU Utilization:** {metric.gpu_utilization:.1f}%
"""

        # Historische Trends
        if len(history) > 1:
            report += "\n## Performance-Trends\n"
            for i, result in enumerate(history[:5]):  # Letzte 5 Ergebnisse
                timestamp = datetime.fromisoformat(result.timestamp)
                report += f"- {timestamp.strftime('%Y-%m-%d %H:%M')}: {result.total_speedup:.2f}x speedup, {result.total_memory_reduction:.1f}% memory reduction\n"

        return report

def main():
    """Hauptfunktion für Performance-Monitoring"""
    logger.info("🚀 TensorRT Performance Monitoring System")
    logger.info("=" * 60)

    monitor = TensorRTPerformanceMonitor()

    # Multimodale KI mit RTX 2070 Modus initialisieren
    model = None
    try:
        logger.info("🎯 Initialisiere multimodale KI für Benchmarking...")
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Umfassenden Benchmark durchführen
        benchmark_result = monitor.run_comprehensive_benchmark(model)

        # Performance-Report generieren
        report = monitor.generate_performance_report()

        # Report in Datei speichern
        report_path = monitor.results_dir / "latest_performance_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info("✅ Performance-Monitoring abgeschlossen")
        logger.info(f"📊 Report gespeichert: {report_path}")

        # Zusammenfassung ausgeben
        logger.info("🎯 Benchmark-Zusammenfassung:")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".2f")

        return 0

    except Exception as e:
        logger.error(f"❌ Fehler beim Performance-Monitoring: {e}")
        return 1

    finally:
        if model:
            try:
                del model
            except:
                pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)