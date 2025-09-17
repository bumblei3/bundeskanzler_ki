#!/usr/bin/env python3
"""
Erweiterte Quantisierungs-Optimierung für Bundeskanzler KI
Automatische Modell-Quantisierung mit Performance-Monitoring
Unterstützt 4-bit, 8-bit und dynamische Quantisierung
"""

import gc
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Konfiguration für Modell-Quantisierung"""
    quantization_type: str  # "4bit", "8bit", "none"
    compute_dtype: torch.dtype
    use_double_quant: bool = True
    quant_type: str = "nf4"  # "nf4" oder "fp4"
    enable_cpu_offload: bool = True
    skip_modules: List[str] = None

    def __post_init__(self):
        if self.skip_modules is None:
            self.skip_modules = ["lm_head"]


@dataclass
class PerformanceMetrics:
    """Performance-Metriken für quantisierte Modelle"""
    model_name: str
    quantization_type: str
    memory_usage_mb: float
    load_time_seconds: float
    inference_time_ms: float
    throughput_tokens_per_sec: float
    model_size_mb: float
    compression_ratio: float


class QuantizationOptimizer:
    """
    Intelligente Quantisierungs-Optimierung für verschiedene GPU-Konfigurationen
    Automatische Auswahl der besten Quantisierung basierend auf Hardware
    """

    def __init__(self, gpu_memory_gb: Optional[float] = None):
        self.gpu_memory_gb = gpu_memory_gb or self._detect_gpu_memory()
        self.cpu_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Performance-Tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.optimal_configs: Dict[str, QuantizationConfig] = {}

        logger.info(f"🧠 QuantizationOptimizer initialisiert - GPU: {self.gpu_memory_gb:.1f}GB, CPU: {self.cpu_memory_gb:.1f}GB")

    def _detect_gpu_memory(self) -> float:
        """Erkennt verfügbaren GPU-Speicher"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0

    def get_optimal_quantization_config(self, model_name: str, model_type: str = "text") -> QuantizationConfig:
        """
        Ermittelt optimale Quantisierungskonfiguration basierend auf Modell und Hardware

        Args:
            model_name: Name/Größe des Modells (z.B. "gpt2-medium", "gpt2-large")
            model_type: Typ des Modells ("text", "vision", "audio")

        Returns:
            Optimale QuantizationConfig
        """
        # Modell-Größen-Mapping (geschätzt in GB für FP16)
        model_sizes = {
            "gpt2": 0.5,
            "gpt2-medium": 1.5,
            "gpt2-large": 6.0,
            "gpt2-xl": 15.0,
            "clip-base": 0.6,
            "siglip-base": 0.9,
            "whisper-base": 0.3,
            "whisper-small": 0.9,
            "whisper-medium": 3.0,
            "whisper-large": 6.0,
        }

        estimated_size = model_sizes.get(model_name, 1.0)

        # RTX 2070 spezifische Optimierungen (8GB VRAM)
        if self.gpu_memory_gb >= 7.5:  # RTX 2070 / ähnliche
            if estimated_size <= 1.0:
                # Kleine Modelle: Keine Quantisierung für beste Qualität
                return QuantizationConfig(
                    quantization_type="none",
                    compute_dtype=torch.float16,
                    use_double_quant=False,
                )
            elif estimated_size <= 3.0:
                # Mittlere Modelle: 8-bit Quantisierung
                return QuantizationConfig(
                    quantization_type="8bit",
                    compute_dtype=torch.float16,
                    use_double_quant=True,
                    enable_cpu_offload=True,
                )
            else:
                # Große Modelle: 4-bit Quantisierung
                return QuantizationConfig(
                    quantization_type="4bit",
                    compute_dtype=torch.float16,
                    use_double_quant=True,
                    quant_type="nf4",
                    enable_cpu_offload=True,
                )

        # Andere GPUs oder weniger Speicher
        elif self.gpu_memory_gb >= 4.0:
            if estimated_size <= 0.8:
                return QuantizationConfig(
                    quantization_type="none",
                    compute_dtype=torch.float16,
                )
            else:
                return QuantizationConfig(
                    quantization_type="8bit",
                    compute_dtype=torch.float16,
                    use_double_quant=True,
                )
        else:
            # Sehr wenig VRAM: Immer 4-bit
            return QuantizationConfig(
                quantization_type="4bit",
                compute_dtype=torch.float16,
                use_double_quant=True,
                quant_type="nf4",
            )

    def create_bitsandbytes_config(self, config: QuantizationConfig) -> Optional[BitsAndBytesConfig]:
        """Erstellt BitsAndBytesConfig aus QuantizationConfig"""
        if config.quantization_type == "none":
            return None

        if config.quantization_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=config.enable_cpu_offload,
                llm_int8_skip_modules=config.skip_modules,
            )
        elif config.quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=config.compute_dtype,
                bnb_4bit_use_double_quant=config.use_double_quant,
                bnb_4bit_quant_type=config.quant_type,
                llm_int8_skip_modules=config.skip_modules,
            )

        return None

    def load_text_model_optimized(self, model_name: str, tokenizer_name: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Lädt Text-Modell mit optimierter Quantisierung

        Args:
            model_name: HuggingFace Modell-Name
            tokenizer_name: Optionale separate Tokenizer-Name

        Returns:
            Tuple von (model, tokenizer)
        """
        start_time = time.time()
        initial_memory = self._get_gpu_memory_usage()

        # Optimale Konfiguration ermitteln
        config = self.get_optimal_quantization_config(model_name, "text")
        bnb_config = self.create_bitsandbytes_config(config)

        logger.info(f"🚀 Lade Text-Modell {model_name} mit {config.quantization_type} Quantisierung")

        try:
            # Tokenizer laden
            tokenizer_name = tokenizer_name or model_name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Modell laden
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=config.compute_dtype if config.quantization_type == "none" else None,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            load_time = time.time() - start_time
            final_memory = self._get_gpu_memory_usage()
            memory_used = final_memory - initial_memory

            # Performance-Metriken sammeln
            metrics = self._benchmark_text_model(model, tokenizer, model_name, config.quantization_type, load_time, memory_used)
            self.performance_history.append(metrics)

            logger.info(f"✅ {model_name} geladen - {config.quantization_type} Quantisierung - {memory_used:.1f}MB - {load_time:.2f}s")

            return model, tokenizer

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
            raise

    def load_vision_model_optimized(self, model_name: str, processor_name: Optional[str] = None) -> Tuple[Union[CLIPModel, SiglipModel], Union[CLIPProcessor, SiglipProcessor]]:
        """
        Lädt Vision-Modell mit optimierter Quantisierung
        """
        start_time = time.time()
        initial_memory = self._get_gpu_memory_usage()

        config = self.get_optimal_quantization_config(model_name, "vision")
        bnb_config = self.create_bitsandbytes_config(config)

        logger.info(f"🎨 Lade Vision-Modell {model_name} mit {config.quantization_type} Quantisierung")

        try:
            processor_name = processor_name or model_name

            # Modell-Typ erkennen
            if "siglip" in model_name.lower():
                processor = SiglipProcessor.from_pretrained(processor_name)
                model = SiglipModel.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    torch_dtype=config.compute_dtype if config.quantization_type == "none" else None,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
            else:  # CLIP
                processor = CLIPProcessor.from_pretrained(processor_name)
                model = CLIPModel.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    torch_dtype=config.compute_dtype if config.quantization_type == "none" else None,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )

            load_time = time.time() - start_time
            final_memory = self._get_gpu_memory_usage()
            memory_used = final_memory - initial_memory

            logger.info(f"✅ {model_name} geladen - {config.quantization_type} Quantisierung - {memory_used:.1f}MB - {load_time:.2f}s")

            return model, processor

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
            raise

    def load_audio_model_optimized(self, model_name: str, processor_name: Optional[str] = None) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
        """
        Lädt Audio-Modell mit optimierter Quantisierung
        """
        start_time = time.time()
        initial_memory = self._get_gpu_memory_usage()

        config = self.get_optimal_quantization_config(model_name, "audio")
        bnb_config = self.create_bitsandbytes_config(config)

        logger.info(f"🎵 Lade Audio-Modell {model_name} mit {config.quantization_type} Quantisierung")

        try:
            processor_name = processor_name or model_name
            processor = WhisperProcessor.from_pretrained(processor_name)

            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=config.compute_dtype if config.quantization_type == "none" else None,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )

            load_time = time.time() - start_time
            final_memory = self._get_gpu_memory_usage()
            memory_used = final_memory - initial_memory

            logger.info(f"✅ {model_name} geladen - {config.quantization_type} Quantisierung - {memory_used:.1f}MB - {load_time:.2f}s")

            return model, processor

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
            raise

    def _benchmark_text_model(self, model, tokenizer, model_name: str, quant_type: str,
                            load_time: float, memory_used: float) -> PerformanceMetrics:
        """Benchmarkt Text-Modell Performance"""
        try:
            # Test-Input
            test_input = "Die Bundeskanzler KI ist ein fortschrittliches System für"
            inputs = tokenizer(test_input, return_tensors="pt").to(self.device)

            # Inference-Zeit messen
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            inference_time = (time.time() - start_time) * 1000  # ms
            generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            throughput = generated_tokens / (inference_time / 1000)  # tokens/sec

            # Modell-Größe schätzen
            model_size = self._estimate_model_size(model)

            return PerformanceMetrics(
                model_name=model_name,
                quantization_type=quant_type,
                memory_usage_mb=memory_used,
                load_time_seconds=load_time,
                inference_time_ms=inference_time,
                throughput_tokens_per_sec=throughput,
                model_size_mb=model_size,
                compression_ratio=1.0,  # Wird später berechnet
            )

        except Exception as e:
            logger.warning(f"Benchmark fehlgeschlagen für {model_name}: {e}")
            return PerformanceMetrics(
                model_name=model_name,
                quantization_type=quant_type,
                memory_usage_mb=memory_used,
                load_time_seconds=load_time,
                inference_time_ms=0.0,
                throughput_tokens_per_sec=0.0,
                model_size_mb=0.0,
                compression_ratio=1.0,
            )

    def _get_gpu_memory_usage(self) -> float:
        """Gibt aktuellen GPU-Speicherverbrauch in MB zurück"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0

    def _estimate_model_size(self, model) -> float:
        """Schätzt Modell-Größe in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024**2)

    def get_performance_report(self) -> Dict[str, any]:
        """Gibt Performance-Bericht zurück"""
        if not self.performance_history:
            return {"message": "Keine Performance-Daten verfügbar"}

        # Beste Konfigurationen finden
        best_memory = min(self.performance_history, key=lambda x: x.memory_usage_mb)
        best_speed = min(self.performance_history, key=lambda x: x.inference_time_ms)
        best_throughput = max(self.performance_history, key=lambda x: x.throughput_tokens_per_sec)

        return {
            "total_models_tested": len(self.performance_history),
            "best_memory_efficiency": {
                "model": best_memory.model_name,
                "quantization": best_memory.quantization_type,
                "memory_mb": best_memory.memory_usage_mb,
            },
            "best_speed": {
                "model": best_speed.model_name,
                "quantization": best_speed.quantization_type,
                "inference_ms": best_speed.inference_time_ms,
            },
            "best_throughput": {
                "model": best_throughput.model_name,
                "quantization": best_throughput.quantization_type,
                "tokens_per_sec": best_throughput.throughput_tokens_per_sec,
            },
            "all_metrics": [vars(m) for m in self.performance_history],
        }

    def optimize_memory_usage(self):
        """Optimiert Speicherverbrauch"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("🧹 Speicher optimiert")


# Globale Instanz für einfachen Zugriff
quantization_optimizer = QuantizationOptimizer()