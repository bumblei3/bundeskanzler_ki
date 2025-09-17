"""
TensorRT Performance Optimizer für RTX 2070
============================================

Optimiert PyTorch/TensorFlow Modelle für maximale GPU-Performance auf RTX 2070.
Verwendet NVIDIA TensorRT für Inference-Optimierung.

Features:
- ONNX Model Export
- TensorRT Engine Building
- FP16/INT8 Quantization
- Dynamic Shape Support
- Performance Benchmarking
"""

import torch
import tensorrt as trt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TensorRTOptimizer:
    """
    TensorRT Optimizer für maximale RTX 2070 Performance
    """

    def __init__(self,
                 workspace_size: int = 1 << 30,  # 1GB
                 max_batch_size: int = 8,
                 fp16_mode: bool = True,
                 int8_mode: bool = False):
        """
        Initialisiere TensorRT Optimizer

        Args:
            workspace_size: Maximale Workspace-Größe für TensorRT
            max_batch_size: Maximale Batch-Größe
            fp16_mode: FP16-Modus aktivieren
            int8_mode: INT8-Modus aktivieren (benötigt Kalibrierung)
        """
        self.workspace_size = workspace_size
        self.max_batch_size = max_batch_size
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode

        # TensorRT Logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Engine Cache
        self.engines: Dict[str, trt.ICudaEngine] = {}
        self.engine_paths: Dict[str, Path] = {}

        logger.info("🎯 TensorRT Optimizer initialisiert")
        logger.info(f"FP16 Mode: {fp16_mode}, INT8 Mode: {int8_mode}")
        logger.info(f"Max Batch Size: {max_batch_size}")

    def create_onnx_model(self,
                         pytorch_model: torch.nn.Module,
                         input_shape: Tuple[int, ...],
                         onnx_path: Union[str, Path],
                         input_names: List[str] = None,
                         output_names: List[str] = None,
                         model_type: str = "transformer") -> bool:
        """
        Konvertiere PyTorch Modell zu ONNX Format

        Args:
            pytorch_model: PyTorch Modell
            input_shape: Input-Shape (ohne Batch-Dimension)
            onnx_path: Pfad für ONNX-Datei
            input_names: Namen der Input-Tensoren
            output_names: Namen der Output-Tensoren
            model_type: Typ des Modells ("transformer", "cnn", etc.)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Modell in Evaluation-Modus setzen
            pytorch_model.eval()

            # Dummy Input basierend auf Modell-Typ erstellen
            if model_type == "transformer":
                # Für Transformer: Verwende LongTensor für Token-IDs
                dummy_input = torch.randint(0, 1000, (1, input_shape[0])).long().cuda()
            else:
                # Für andere Modelle: FloatTensor
                dummy_input = torch.randn(1, *input_shape).cuda()

            # ONNX Export
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,  # Erhöht für bessere Transformer-Unterstützung
                do_constant_folding=True,
                input_names=input_names or ['input'],
                output_names=output_names or ['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

            logger.info(f"✅ ONNX Modell erstellt: {onnx_path}")
            return True

        except Exception as e:
            logger.error(f"❌ ONNX Export fehlgeschlagen: {e}")
            return False

    def build_tensorrt_engine(self,
                             onnx_path: Union[str, Path],
                             engine_path: Union[str, Path],
                             max_batch_size: int = None) -> Optional[trt.ICudaEngine]:
        """
        Baue TensorRT Engine aus ONNX Modell

        Args:
            onnx_path: Pfad zur ONNX-Datei
            engine_path: Pfad für Engine-Datei
            max_batch_size: Maximale Batch-Größe

        Returns:
            TensorRT Engine oder None bei Fehler
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size

        try:
            # Builder erstellen
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.trt_logger)

            # ONNX Modell parsen
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    logger.error("❌ ONNX Parsing fehlgeschlagen")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None

            # Builder Config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

            # Optimierungen aktivieren
            if self.fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("🎯 FP16 Modus aktiviert")

            if self.int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("🎯 INT8 Modus aktiviert")

            # Profile für dynamische Shapes
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            min_shape = (1,) + input_tensor.shape[1:]
            opt_shape = (max_batch_size // 2,) + input_tensor.shape[1:]
            max_shape = (max_batch_size,) + input_tensor.shape[1:]

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            # Engine bauen
            logger.info("🏗️ Baue TensorRT Engine...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                logger.error("❌ Engine-Building fehlgeschlagen")
                return None

            # Engine serialisieren
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

            logger.info(f"✅ TensorRT Engine erstellt: {engine_path}")
            return serialized_engine

        except Exception as e:
            logger.error(f"❌ Engine-Building fehlgeschlagen: {e}")
            return None

    def load_engine(self, engine_path: Union[str, Path]) -> Optional[trt.ICudaEngine]:
        """
        Lade TensorRT Engine aus Datei

        Args:
            engine_path: Pfad zur Engine-Datei

        Returns:
            TensorRT Engine oder None bei Fehler
        """
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)

            if engine is None:
                logger.error(f"❌ Engine konnte nicht geladen werden: {engine_path}")
                return None

            logger.info(f"✅ Engine geladen: {engine_path}")
            return engine

        except Exception as e:
            logger.error(f"❌ Engine-Loading fehlgeschlagen: {e}")
            return None

    def create_execution_context(self, engine: trt.ICudaEngine) -> Optional[trt.IExecutionContext]:
        """
        Erstelle Execution Context für Engine

        Args:
            engine: TensorRT Engine

        Returns:
            Execution Context oder None bei Fehler
        """
        try:
            context = engine.create_execution_context()
            logger.info("✅ Execution Context erstellt")
            return context
        except Exception as e:
            logger.error(f"❌ Execution Context fehlgeschlagen: {e}")
            return None

    def benchmark_inference(self,
                           engine: trt.ICudaEngine,
                           input_shape: Tuple[int, ...],
                           num_runs: int = 100,
                           warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark TensorRT Inference Performance

        Args:
            engine: TensorRT Engine
            input_shape: Input-Shape für Test
            num_runs: Anzahl der Benchmark-Runs
            warmup_runs: Anzahl der Warmup-Runs

        Returns:
            Performance-Metriken
        """
        try:
            context = self.create_execution_context(engine)
            if context is None:
                return {}

            # GPU Buffers vorbereiten
            input_tensor = engine.get_tensor_name(0)
            output_tensor = engine.get_tensor_name(1)

            # Dummy Input
            batch_size = 1
            input_data = np.random.randn(batch_size, *input_shape).astype(np.float32)

            # GPU Memory allokieren
            input_buf = torch.from_numpy(input_data).cuda()
            output_buf = torch.zeros((batch_size, engine.get_tensor_shape(output_tensor)[1])).cuda()

            # Bindings
            bindings = [input_buf.data_ptr(), output_buf.data_ptr()]

            # Warmup
            logger.info(f"🔥 Warmup ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                context.execute_v2(bindings)

            # Benchmark
            logger.info(f"📊 Benchmark ({num_runs} runs)...")
            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(num_runs):
                context.execute_v2(bindings)

            torch.cuda.synchronize()
            end_time = time.time()

            # Metriken berechnen
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            throughput = num_runs / total_time

            metrics = {
                'avg_inference_time_ms': avg_time * 1000,
                'throughput_inferences_per_sec': throughput,
                'total_time_sec': total_time,
                'num_runs': num_runs
            }

            logger.info("📈 Benchmark Ergebnisse:")
            logger.info(".2f")
            logger.info(".1f")

            return metrics

        except Exception as e:
            logger.error(f"❌ Benchmark fehlgeschlagen: {e}")
            return {}

    def optimize_pytorch_model(self,
                              model: torch.nn.Module,
                              input_shape: Tuple[int, ...],
                              model_name: str,
                              cache_dir: Union[str, Path] = "./tensorrt_cache") -> Optional[Dict]:
        """
        Vollständige Optimierung eines PyTorch Modells

        Args:
            model: PyTorch Modell
            input_shape: Input-Shape (ohne Batch-Dimension)
            model_name: Name für Cache-Dateien
            cache_dir: Cache-Verzeichnis

        Returns:
            Optimierungsergebnisse oder None bei Fehler
        """
        try:
            cache_path = Path(cache_dir)
            cache_path.mkdir(exist_ok=True)

            onnx_path = cache_path / f"{model_name}.onnx"
            engine_path = cache_path / f"{model_name}.engine"

            # Schritt 1: ONNX Export
            logger.info("📤 Schritt 1: ONNX Export...")
            if not self.create_onnx_model(model, input_shape, onnx_path, model_type="transformer"):
                return None

            # Schritt 2: TensorRT Engine bauen
            logger.info("🏗️ Schritt 2: TensorRT Engine bauen...")
            engine = self.build_tensorrt_engine(onnx_path, engine_path)
            if engine is None:
                return None

            # Schritt 3: Benchmark
            logger.info("📊 Schritt 3: Performance Benchmark...")
            # Engine für Benchmarking laden
            loaded_engine = self.load_engine(engine_path)
            if loaded_engine is None:
                return None
            metrics = self.benchmark_inference(loaded_engine, input_shape)

            if not metrics:
                return None

            # Ergebnisse
            result = {
                'model_name': model_name,
                'onnx_path': str(onnx_path),
                'engine_path': str(engine_path),
                'engine': engine,
                'metrics': metrics,
                'optimization_type': 'FP16' if self.fp16_mode else 'FP32',
                'timestamp': time.time()
            }

            logger.info("🎉 Modell erfolgreich optimiert!")
            logger.info(f"📁 Engine gespeichert: {engine_path}")

            return result

        except Exception as e:
            logger.error(f"❌ Modell-Optimierung fehlgeschlagen: {e}")
            return None

class PerformanceMonitor:
    """
    Überwacht und analysiert Performance-Metriken
    """

    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, metadata: Dict = None):
        """Zeichne Metrik auf"""
        metric = {
            'timestamp': time.time(),
            'name': name,
            'value': value,
            'metadata': metadata or {}
        }
        self.metrics_history.append(metric)

    def get_summary(self) -> Dict:
        """Erstelle Performance-Zusammenfassung"""
        if not self.metrics_history:
            return {}

        # Gruppiere nach Metrik-Namen
        metric_groups = {}
        for metric in self.metrics_history:
            name = metric['name']
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric['value'])

        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return summary

    def print_summary(self):
        """Drucke Performance-Zusammenfassung"""
        summary = self.get_summary()
        if not summary:
            print("Keine Metriken verfügbar")
            return

        print("📊 Performance-Zusammenfassung:")
        print("=" * 50)

        for name, stats in summary.items():
            print(f"\n🔹 {name}:")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"   Anzahl: {stats['count']}")

# Convenience Functions
def optimize_for_rtx2070(model: torch.nn.Module,
                        input_shape: Tuple[int, ...],
                        model_name: str) -> Optional[Dict]:
    """
    Optimierung für RTX 2070 mit Standardeinstellungen

    Args:
        model: PyTorch Modell
        input_shape: Input-Shape
        model_name: Modell-Name

    Returns:
        Optimierungsergebnisse
    """
    optimizer = TensorRTOptimizer(
        fp16_mode=True,
        int8_mode=False,
        max_batch_size=4
    )

    return optimizer.optimize_pytorch_model(model, input_shape, model_name)

if __name__ == "__main__":
    # Beispiel für RTX 2070 Optimierung
    print("🚀 TensorRT Performance Optimizer für RTX 2070")
    print("=" * 60)

    # Teste GPU-Verfügbarkeit
    if torch.cuda.is_available():
        print("✅ CUDA verfügbar")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(".1f")
    else:
        print("❌ CUDA nicht verfügbar")
        exit(1)

    print("\n🎯 TensorRT Optimizer bereit für Performance-Maximierung!")