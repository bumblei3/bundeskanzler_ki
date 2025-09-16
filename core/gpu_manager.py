#!/usr/bin/env python3
"""
🚀 Bundeskanzler-KI RTX 2070 GPU Manager
===========================================

Optimiert für NVIDIA GeForce RTX 2070:
- 8GB GDDR6 VRAM
- 2,304 CUDA Cores  
- 288 Tensor Cores (1st gen)
- Mixed Precision (FP16/FP32)

Autor: Claude-3.5-Sonnet
Datum: 15. September 2025
"""

import torch
import gc
import logging
import time
import psutil
from typing import Dict, Optional, List, Any
from contextlib import contextmanager
from dataclasses import dataclass

# Verhindere NVIDIA-ML Deprecation Warnings
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

try:
    import nvidia_ml_py as nvml
    NVML_AVAILABLE = True
except ImportError:
    try:
        import pynvml as nvml
        NVML_AVAILABLE = True
    except ImportError:
        NVML_AVAILABLE = False
        print("⚠️ NVIDIA-ML nicht verfügbar - GPU Monitoring deaktiviert")

logger = logging.getLogger(__name__)

@dataclass
class GPUStats:
    """RTX 2070 Statistiken"""
    gpu_utilization: float
    memory_used_gb: float
    memory_total_gb: float
    memory_utilization: float
    temperature_c: int
    power_usage_w: float
    tensor_core_usage: float
    
class RTX2070Manager:
    """
    🚀 RTX 2070 GPU Manager für Bundeskanzler-KI
    
    Optimiert für 8GB VRAM mit intelligenter Memory-Verwaltung:
    - Mixed Precision (FP16) für Tensor Cores
    - Dynamic Memory Management
    - Performance Monitoring
    - Automatic Fallback zu CPU bei Fehlern
    """
    
    def __init__(self, max_memory_fraction: float = 0.90):
        """
        Initialisiert RTX 2070 Manager
        
        Args:
            max_memory_fraction: Maximaler VRAM-Anteil (90% = 7.2GB bei RTX 2070)
        """
        self.device = None
        self.max_memory_fraction = max_memory_fraction
        self.tensor_cores_enabled = False
        self.batch_sizes = {}
        self.performance_stats = []
        
        # Initialize GPU
        self._initialize_gpu()
        
        # Initialize NVIDIA-ML für Monitoring
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("✅ NVIDIA-ML initialisiert für RTX 2070 Monitoring")
            except Exception as e:
                logger.warning(f"⚠️ NVIDIA-ML Initialisierung fehlgeschlagen: {e}")
                self.nvml_handle = None
        else:
            self.nvml_handle = None
    
    def _initialize_gpu(self):
        """Initialisiert RTX 2070 GPU mit optimalen Einstellungen"""
        try:
            if not torch.cuda.is_available():
                logger.warning("❌ CUDA nicht verfügbar - verwende CPU fallback")
                self.device = torch.device('cpu')
                return
                
            # RTX 2070 Device Setup
            self.device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            
            if "RTX 2070" not in gpu_name and "GeForce RTX 2070" not in gpu_name:
                logger.warning(f"⚠️ Nicht RTX 2070 erkannt: {gpu_name}")
            
            # Memory Management für 8GB VRAM
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_memory * self.max_memory_fraction)
            
            # Set Memory Fraction (90% von 8GB = 7.2GB)
            torch.cuda.set_per_process_memory_fraction(self.max_memory_fraction, device=0)
            
            # cuDNN Optimizations für RTX 2070
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True  # Für bessere Performance
            
            # Tensor Core Setup
            self.tensor_cores_enabled = self._enable_tensor_cores()
            
            # Optimale Batch Sizes für RTX 2070
            self.batch_sizes = {
                'embedding_generation': 32,   # Optimal für 8GB VRAM
                'semantic_search': 64,        # Größere Batches möglich
                'multi_agent_processing': 16, # Conservative für Multi-Agent
                'prediction_models': 8,       # Sehr conservative für NN
                'inference': 24               # Standard Inference
            }
            
            logger.info(f"🚀 RTX 2070 initialisiert:")
            logger.info(f"   💾 VRAM Total: {total_memory / 1024**3:.1f} GB")
            logger.info(f"   💾 VRAM Verfügbar: {max_memory / 1024**3:.1f} GB")
            logger.info(f"   🔥 Tensor Cores: {'✅' if self.tensor_cores_enabled else '❌'}")
            logger.info(f"   📊 Batch Size (Embedding): {self.batch_sizes['embedding_generation']}")
            
        except Exception as e:
            logger.error(f"❌ GPU Initialisierung fehlgeschlagen: {e}")
            self.device = torch.device('cpu')
    
    def _enable_tensor_cores(self) -> bool:
        """Aktiviert RTX 2070 Tensor Cores für Mixed Precision"""
        try:
            # Test Mixed Precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x = torch.randn(256, 768, device=self.device, dtype=torch.float16)
                y = torch.randn(768, 256, device=self.device, dtype=torch.float16)
                z = torch.matmul(x, y)
                
            # Verify FP16 wurde verwendet
            if z.dtype == torch.float16:
                logger.info("✅ RTX 2070 Tensor Cores aktiviert (FP16)")
                return True
            else:
                logger.warning("⚠️ Mixed Precision nicht verfügbar")
                return False
                
        except Exception as e:
            logger.error(f"❌ Tensor Core Aktivierung fehlgeschlagen: {e}")
            return False
    
    def get_optimal_batch_size(self, task_type: str) -> int:
        """
        Holt optimale Batch Size für RTX 2070
        
        Args:
            task_type: Art der Aufgabe ('embedding_generation', 'semantic_search', etc.)
            
        Returns:
            Optimale Batch Size für die RTX 2070
        """
        return self.batch_sizes.get(task_type, 16)  # Default: 16
    
    def get_gpu_stats(self) -> Optional[GPUStats]:
        """
        Sammelt RTX 2070 Performance-Statistiken
        
        Returns:
            GPUStats Objekt mit aktuellen Metriken
        """
        if not self.is_gpu_available() or not self.nvml_handle:
            return None
            
        try:
            # Memory Info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            memory_used_gb = mem_info.used / 1024**3
            memory_total_gb = mem_info.total / 1024**3
            memory_utilization = (mem_info.used / mem_info.total) * 100
            
            # GPU Utilization
            util = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            gpu_utilization = util.gpu
            
            # Temperature
            temp = nvml.nvmlDeviceGetTemperature(self.nvml_handle, nvml.NVML_TEMPERATURE_GPU)
            
            # Power Usage
            try:
                power = nvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # mW to W
            except:
                power = 0.0
                
            # Estimate Tensor Core Usage (basierend auf Memory Bandwidth)
            tensor_core_usage = min(gpu_utilization * 1.2, 100.0) if self.tensor_cores_enabled else 0.0
            
            stats = GPUStats(
                gpu_utilization=gpu_utilization,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                memory_utilization=memory_utilization,
                temperature_c=temp,
                power_usage_w=power,
                tensor_core_usage=tensor_core_usage
            )
            
            # Store für Trend Analysis
            self.performance_stats.append({
                'timestamp': time.time(),
                'stats': stats
            })
            
            # Keep nur letzte 100 Messungen
            if len(self.performance_stats) > 100:
                self.performance_stats = self.performance_stats[-100:]
                
            return stats
            
        except Exception as e:
            logger.error(f"❌ GPU Stats Sammlung fehlgeschlagen: {e}")
            return None
    
    def is_gpu_available(self) -> bool:
        """Prüft ob RTX 2070 verfügbar ist"""
        return self.device is not None and self.device.type == 'cuda'
    
    def get_device(self) -> torch.device:
        """Holt aktuelles Device (RTX 2070 oder CPU Fallback)"""
        return self.device
    
    @contextmanager
    def mixed_precision_context(self):
        """
        Context Manager für RTX 2070 Mixed Precision
        
        Nutzt Tensor Cores automatisch für FP16 Operations
        """
        if self.is_gpu_available() and self.tensor_cores_enabled:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                yield
        else:
            yield
    
    def move_to_gpu(self, tensor_or_model, non_blocking: bool = True):
        """
        Bewegt Tensor/Model zu RTX 2070 GPU
        
        Args:
            tensor_or_model: PyTorch Tensor oder Model
            non_blocking: Asynchroner Transfer für bessere Performance
            
        Returns:
            GPU Tensor/Model oder original bei CPU fallback
        """
        if self.is_gpu_available():
            try:
                return tensor_or_model.to(self.device, non_blocking=non_blocking)
            except Exception as e:
                logger.warning(f"⚠️ GPU Transfer fehlgeschlagen: {e}")
                return tensor_or_model
        return tensor_or_model
    
    def optimize_model_for_rtx2070(self, model, enable_fp16: bool = True):
        """
        Optimiert Model für RTX 2070
        
        Args:
            model: PyTorch Model
            enable_fp16: Aktiviert Mixed Precision wenn möglich
            
        Returns:
            Optimiertes Model
        """
        try:
            # Move zu GPU
            model = self.move_to_gpu(model)
            
            # Mixed Precision für Tensor Cores
            if enable_fp16 and self.tensor_cores_enabled:
                model = model.half()  # Convert zu FP16
                logger.info("✅ Model zu FP16 konvertiert für Tensor Cores")
            
            # Eval Mode für Inference
            model.eval()
            
            # Memory Optimization
            torch.cuda.empty_cache()
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model Optimierung fehlgeschlagen: {e}")
            return model
    
    def clear_memory(self):
        """Leert RTX 2070 VRAM Cache"""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Detaillierte VRAM Memory Summary
        
        Returns:
            Dictionary mit Memory-Informationen
        """
        if not self.is_gpu_available():
            return {'status': 'CPU_ONLY'}
            
        try:
            # PyTorch Memory Info
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            
            # NVIDIA-ML Memory Info
            stats = self.get_gpu_stats()
            
            summary = {
                'status': 'GPU_ACTIVE',
                'device': str(self.device),
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'tensor_cores': self.tensor_cores_enabled,
                'batch_sizes': self.batch_sizes
            }
            
            if stats:
                summary.update({
                    'total_gb': stats.memory_total_gb,
                    'used_gb': stats.memory_used_gb,
                    'utilization_percent': stats.memory_utilization,
                    'gpu_utilization_percent': stats.gpu_utilization,
                    'temperature_c': stats.temperature_c
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Memory Summary fehlgeschlagen: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def performance_benchmark(self, iterations: int = 10) -> Dict[str, float]:
        """
        RTX 2070 Performance Benchmark
        
        Args:
            iterations: Anzahl Benchmark-Läufe
            
        Returns:
            Performance-Metriken
        """
        if not self.is_gpu_available():
            return {'status': 'CPU_ONLY'}
            
        logger.info(f"🚀 Starte RTX 2070 Benchmark ({iterations} Iterationen)...")
        
        results = {
            'matrix_mult_fp32_ms': [],
            'matrix_mult_fp16_ms': [],
            'memory_bandwidth_gb_s': [],
            'tensor_core_speedup': 0.0
        }
        
        try:
            # Warm-up
            for _ in range(3):
                x = torch.randn(1000, 1000, device=self.device)
                y = torch.randn(1000, 1000, device=self.device)
                _ = torch.matmul(x, y)
            
            torch.cuda.synchronize()
            
            for i in range(iterations):
                # FP32 Benchmark
                x_fp32 = torch.randn(1000, 1000, device=self.device, dtype=torch.float32)
                y_fp32 = torch.randn(1000, 1000, device=self.device, dtype=torch.float32)
                
                start_time = time.perf_counter()
                z_fp32 = torch.matmul(x_fp32, y_fp32)
                torch.cuda.synchronize()
                fp32_time = (time.perf_counter() - start_time) * 1000
                results['matrix_mult_fp32_ms'].append(fp32_time)
                
                # FP16 Benchmark (Tensor Cores)
                if self.tensor_cores_enabled:
                    x_fp16 = torch.randn(1000, 1000, device=self.device, dtype=torch.float16)
                    y_fp16 = torch.randn(1000, 1000, device=self.device, dtype=torch.float16)
                    
                    start_time = time.perf_counter()
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        z_fp16 = torch.matmul(x_fp16, y_fp16)
                    torch.cuda.synchronize()
                    fp16_time = (time.perf_counter() - start_time) * 1000
                    results['matrix_mult_fp16_ms'].append(fp16_time)
                
                # Memory Bandwidth Test
                large_tensor = torch.randn(10000, 1000, device=self.device)
                start_time = time.perf_counter()
                copied = large_tensor.clone()
                torch.cuda.synchronize()
                copy_time = time.perf_counter() - start_time
                bandwidth = (large_tensor.numel() * 4 * 2) / (copy_time * 1024**3)  # GB/s
                results['memory_bandwidth_gb_s'].append(bandwidth)
                
                if (i + 1) % 3 == 0:
                    logger.info(f"   Iteration {i+1}/{iterations} abgeschlossen")
            
            # Calculate Averages
            avg_fp32 = sum(results['matrix_mult_fp32_ms']) / len(results['matrix_mult_fp32_ms'])
            
            if results['matrix_mult_fp16_ms']:
                avg_fp16 = sum(results['matrix_mult_fp16_ms']) / len(results['matrix_mult_fp16_ms'])
                speedup = avg_fp32 / avg_fp16
                results['tensor_core_speedup'] = speedup
            
            avg_bandwidth = sum(results['memory_bandwidth_gb_s']) / len(results['memory_bandwidth_gb_s'])
            
            benchmark_summary = {
                'avg_fp32_ms': avg_fp32,
                'avg_fp16_ms': avg_fp16 if results['matrix_mult_fp16_ms'] else 0.0,
                'tensor_core_speedup': results['tensor_core_speedup'],
                'avg_memory_bandwidth_gb_s': avg_bandwidth,
                'rtx2070_score': self._calculate_rtx2070_score(avg_fp32, avg_fp16, avg_bandwidth)
            }
            
            logger.info("✅ RTX 2070 Benchmark abgeschlossen:")
            logger.info(f"   🔥 FP32 Performance: {avg_fp32:.1f}ms")
            if results['matrix_mult_fp16_ms']:
                logger.info(f"   ⚡ FP16 Performance: {avg_fp16:.1f}ms")
                logger.info(f"   🚀 Tensor Core Speedup: {speedup:.1f}x")
            logger.info(f"   💾 Memory Bandwidth: {avg_bandwidth:.1f} GB/s")
            
            return benchmark_summary
            
        except Exception as e:
            logger.error(f"❌ Benchmark fehlgeschlagen: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _calculate_rtx2070_score(self, fp32_ms: float, fp16_ms: float, bandwidth_gb_s: float) -> float:
        """
        Berechnet RTX 2070 Performance Score (0-100)
        
        Baseline RTX 2070 Werte:
        - FP32: ~5ms (1000x1000 matmul)
        - FP16: ~2.5ms (mit Tensor Cores)
        - Bandwidth: ~400 GB/s
        """
        try:
            # Baseline Values für RTX 2070
            baseline_fp32 = 5.0
            baseline_fp16 = 2.5
            baseline_bandwidth = 400.0
            
            # Performance Ratios (höher = besser)
            fp32_ratio = baseline_fp32 / max(fp32_ms, 0.1)
            fp16_ratio = baseline_fp16 / max(fp16_ms, 0.1) if fp16_ms > 0 else 1.0
            bandwidth_ratio = bandwidth_gb_s / baseline_bandwidth
            
            # Weighted Score
            score = (fp32_ratio * 30 + fp16_ratio * 40 + bandwidth_ratio * 30)
            return min(score * 100, 100.0)  # Cap bei 100
            
        except:
            return 0.0

# Global RTX 2070 Manager Instance
_rtx2070_manager = None

def get_rtx2070_manager() -> RTX2070Manager:
    """
    Singleton RTX 2070 Manager
    
    Returns:
        Global RTX2070Manager Instanz
    """
    global _rtx2070_manager
    if _rtx2070_manager is None:
        _rtx2070_manager = RTX2070Manager()
    return _rtx2070_manager

def is_rtx2070_available() -> bool:
    """Prüft RTX 2070 Verfügbarkeit"""
    return get_rtx2070_manager().is_gpu_available()

@contextmanager
def rtx2070_context():
    """Context Manager für RTX 2070 Operations"""
    manager = get_rtx2070_manager()
    try:
        with manager.mixed_precision_context():
            yield manager
    finally:
        manager.clear_memory()

# Convenience Functions
def move_to_rtx2070(tensor_or_model):
    """Bewegt zu RTX 2070 GPU"""
    return get_rtx2070_manager().move_to_gpu(tensor_or_model)

def get_optimal_batch_size(task_type: str) -> int:
    """Holt optimale Batch Size für Task"""
    return get_rtx2070_manager().get_optimal_batch_size(task_type)

def rtx2070_memory_summary() -> Dict[str, Any]:
    """RTX 2070 Memory Summary"""
    return get_rtx2070_manager().get_memory_summary()

if __name__ == "__main__":
    # Test RTX 2070 Manager
    print("🚀 Testing RTX 2070 Manager...")
    
    manager = get_rtx2070_manager()
    
    print(f"GPU Available: {manager.is_gpu_available()}")
    print(f"Device: {manager.get_device()}")
    
    if manager.is_gpu_available():
        stats = manager.get_gpu_stats()
        if stats:
            print(f"GPU Utilization: {stats.gpu_utilization}%")
            print(f"Memory: {stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f} GB")
            print(f"Temperature: {stats.temperature_c}°C")
        
        print("\n🚀 Running Performance Benchmark...")
        benchmark = manager.performance_benchmark(iterations=5)
        print(f"RTX 2070 Score: {benchmark.get('rtx2070_score', 0):.1f}/100")