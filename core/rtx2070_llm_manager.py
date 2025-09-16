#!/usr/bin/env python3
"""
🚀 RTX 2070-Optimierte LLM-Integration für Bundeskanzler-KI
===========================================================

Optimiert für NVIDIA GeForce RTX 2070 (8GB VRAM):
- Mistral 7B / Llama 2 7B (4-bit quantisiert)
- GPU-optimierte Inferenz
- Dynamic Model Loading basierend auf VRAM
- Fallback auf CPU bei Bedarf

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

logger = logging.getLogger(__name__)


@dataclass
class RTX2070ModelConfig:
    """RTX 2070 optimierte Modell-Konfigurationen"""

    name: str
    huggingface_id: str
    max_memory_gb: float
    context_length: int
    quantization: str = "4bit"
    description: str = ""

    def can_run_on_rtx2070(self, available_vram_gb: float) -> bool:
        """Prüft ob Modell auf verfügbarem VRAM läuft"""
        return available_vram_gb >= self.max_memory_gb


# RTX 2070 optimierte Modelle (getestet für 8GB VRAM)
RTX2070_MODELS = {
    "mistral_7b": RTX2070ModelConfig(
        name="Mistral 7B Instruct",
        huggingface_id="mistralai/Mistral-7B-Instruct-v0.1",
        max_memory_gb=4.2,
        context_length=4096,
        description="Ausgewogene Leistung für deutsche Texte",
    ),
    "llama2_7b": RTX2070ModelConfig(
        name="Llama 2 7B Chat",
        huggingface_id="meta-llama/Llama-2-7b-chat-hf",
        max_memory_gb=4.0,
        context_length=4096,
        description="Meta's optimiertes Chat-Modell",
    ),
    "german_gpt2": RTX2070ModelConfig(
        name="German GPT-2 Large",
        huggingface_id="dbmdz/german-gpt2",
        max_memory_gb=1.5,
        context_length=1024,
        quantization="none",
        description="Schnell für einfache Aufgaben",
    ),
}


class RTX2070LLMManager:
    """
    RTX 2070 optimierter LLM-Manager mit Dynamic Model Loading
    """

    def __init__(self, gpu_manager=None):
        self.gpu_manager = gpu_manager
        self.current_model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # RTX 2070 spezifische Optimierungen
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def get_available_vram_gb(self) -> float:
        """Ermittelt verfügbaren VRAM in GB"""
        if self.gpu_manager and hasattr(self.gpu_manager, "get_gpu_stats"):
            stats = self.gpu_manager.get_gpu_stats()
            return stats.memory_total_gb - stats.memory_used_gb
        elif torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0

    def select_optimal_model(self, task_complexity: str = "medium") -> str:
        """
        Wählt optimales Modell basierend auf Task-Komplexität und VRAM
        """
        available_vram = self.get_available_vram_gb()

        if task_complexity == "simple":
            # Für einfache Aufgaben: German GPT-2 (schnell, wenig VRAM)
            if RTX2070_MODELS["german_gpt2"].can_run_on_rtx2070(available_vram):
                return "german_gpt2"
        elif task_complexity == "complex":
            # Für komplexe Aufgaben: Mistral 7B wenn möglich
            if RTX2070_MODELS["mistral_7b"].can_run_on_rtx2070(available_vram):
                return "mistral_7b"

        # Fallback: Llama 2 7B (meist kompatibel)
        if RTX2070_MODELS["llama2_7b"].can_run_on_rtx2070(available_vram):
            return "llama2_7b"

        # Notfall-Fallback: CPU-Modell
        return "cpu_fallback"

    def load_model(self, model_key: str) -> bool:
        """
        Lädt Modell mit RTX 2070 Optimierungen
        """
        try:
            if model_key not in RTX2070_MODELS and model_key != "cpu_fallback":
                logger.error(f"Unbekanntes Modell: {model_key}")
                return False

            # Entlade aktuelles Modell
            self.unload_model()

            if model_key == "cpu_fallback":
                # Einfaches CPU-Modell für Notfälle
                logger.info("Fallback auf CPU-Modell")
                return self._load_cpu_fallback()
            else:
                config = RTX2070_MODELS[model_key]
                logger.info(f"Lade RTX 2070-optimiertes Modell: {config.name}")

                # RTX 2070 spezifische Lade-Konfiguration
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                }

                if config.quantization == "4bit":
                    model_kwargs["quantization_config"] = self.quantization_config

                # Modell laden
                self.tokenizer = AutoTokenizer.from_pretrained(config.huggingface_id)
                model = AutoModelForCausalLM.from_pretrained(config.huggingface_id, **model_kwargs)

                # Pipeline für Text-Generation erstellen
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                self.current_model = model_key
                logger.info(f"✅ Modell {config.name} erfolgreich geladen")
                return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {model_key}: {e}")
            return False

    def _load_cpu_fallback(self) -> bool:
        """Lädt einfaches CPU-Modell für Notfälle"""
        try:
            from transformers import pipeline as cpu_pipeline

            self.pipeline = cpu_pipeline(
                "text-generation",
                model="dbmdz/german-gpt2",
                device=-1,  # CPU
                max_new_tokens=256,
                temperature=0.8,
            )
            self.current_model = "cpu_fallback"
            logger.info("✅ CPU-Fallback-Modell geladen")
            return True
        except Exception as e:
            logger.error(f"❌ CPU-Fallback fehlgeschlagen: {e}")
            return False

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        """
        Generiert Antwort mit aktuellem Modell
        """
        if not self.pipeline:
            return "❌ Kein Modell geladen"

        try:
            # Kontext in Prompt integrieren
            full_prompt = (
                f"Kontext: {context}\n\nFrage: {prompt}\n\nAntwort:" if context else prompt
            )

            # RTX 2070 optimierte Generation
            with torch.no_grad():
                outputs = self.pipeline(
                    full_prompt,
                    max_new_tokens=kwargs.get("max_new_tokens", 256),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else 50256,
                    **kwargs,
                )

            # Antwort extrahieren
            generated_text = outputs[0]["generated_text"]
            response = generated_text.replace(full_prompt, "").strip()

            return response

        except Exception as e:
            logger.error(f"❌ Fehler bei der Generierung: {e}")
            return f"❌ Generierungsfehler: {str(e)}"

    def unload_model(self):
        """Entlädt aktuelles Modell zur VRAM-Freigabe"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if self.current_model:
            logger.info(f"🗑️ Modell {self.current_model} entladen")
            self.current_model = None

        # GPU Memory freigeben
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_system_info(self) -> Dict[str, Any]:
        """Gibt System-Informationen zurück"""
        available_vram = self.get_available_vram_gb()
        available_models = []

        for key, config in RTX2070_MODELS.items():
            if config.can_run_on_rtx2070(available_vram):
                available_models.append(
                    {
                        "key": key,
                        "name": config.name,
                        "max_memory_gb": config.max_memory_gb,
                        "description": config.description,
                    }
                )

        return {
            "gpu_memory_gb": available_vram,
            "device": self.device,
            "current_model": self.current_model,
            "available_models": available_models,
            "quantization_enabled": True,
            "cuda_available": torch.cuda.is_available(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über aktuelles Modell zurück"""
        available_vram = self.get_available_vram_gb()

        if self.current_model and self.current_model in RTX2070_MODELS:
            config = RTX2070_MODELS[self.current_model]
            return {
                "model": config.name,
                "huggingface_id": config.huggingface_id,
                "max_memory_gb": config.max_memory_gb,
                "available_vram_gb": available_vram,
                "context_length": config.context_length,
                "device": self.device,
            }
        else:
            return {
                "model": self.current_model or "none",
                "available_vram_gb": available_vram,
                "device": self.device,
            }


# Globale Instanz für einfachen Zugriff
_rtx2070_llm_manager = None


def get_rtx2070_llm_manager(gpu_manager=None) -> RTX2070LLMManager:
    """Factory-Funktion für RTX 2070 LLM Manager"""
    global _rtx2070_llm_manager
    if _rtx2070_llm_manager is None:
        _rtx2070_llm_manager = RTX2070LLMManager(gpu_manager)
    return _rtx2070_llm_manager


# Kompatibilitätsfunktionen für bestehende Codebasis
def generate_llm_response(
    prompt: str, context: str = "", model_preference: str = None, max_tokens: int = 100
) -> str:
    """
    Kompatibilitätsfunktion für bestehende generate_response Aufrufe
    """
    manager = get_rtx2070_llm_manager()

    # Modell basierend auf Prompt-Komplexität wählen
    if model_preference:
        model_key = model_preference
    else:
        # Automatische Modell-Auswahl
        complexity = "complex" if len(prompt.split()) > 20 else "simple"
        model_key = manager.select_optimal_model(complexity)

    # Modell laden falls nötig
    if manager.current_model != model_key:
        if not manager.load_model(model_key):
            return "❌ Modell konnte nicht geladen werden"

    return manager.generate_response(prompt, context, max_new_tokens=max_tokens)


if __name__ == "__main__":
    # Test der RTX 2070 LLM Integration
    manager = get_rtx2070_llm_manager()

    print("🚀 RTX 2070 LLM Manager Test")
    print(f"Verfügbarer VRAM: {manager.get_available_vram_gb():.1f} GB")

    # Optimales Modell wählen
    model_key = manager.select_optimal_model("medium")
    print(f"Ausgewähltes Modell: {model_key}")

    # Modell laden
    if manager.load_model(model_key):
        # Test-Generierung
        test_prompt = "Was ist die Bedeutung der Energiewende für Deutschland?"
        response = manager.generate_response(test_prompt)
        print(f"\nTest-Antwort:\n{response}")

        # Modell-Info anzeigen
        info = manager.get_model_info()
        print(f"\nModell-Info: {info}")
    else:
        print("❌ Modell konnte nicht geladen werden")
