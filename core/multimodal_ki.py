#!/usr/bin/env python3
"""
Erweiterte Bundeskanzler KI mit multimodalen Fähigkeiten
Unterstützt Text, Bilder, Audio und Video für umfassende Interaktion
MIT UPGRADE-OPTIONEN: GPT-4, Claude, LLaVA, SigLIP, Whisper Large
"""

import gc
import io
import logging
import os
from typing import Dict, List, Optional, Union

import anthropic
import cv2
import librosa
import numpy as np
import requests
import tensorflow as tf
import torch
from openai import OpenAI
from PIL import Image
from tensorflow.keras import layers
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
    pipeline,
)

logger = logging.getLogger(__name__)


class MultimodalTransformerModel:
    """
    Multimodales Transformer-Modell für Text, Bilder, Audio und Video
    MIT SUPPORT FÜR FORTSCHRITTLICHERE MODELLE
    """

    def __init__(self, model_tier: str = "rtx2070"):
        """
        Args:
            model_tier: "rtx2070" (RTX 2070 optimiert), "basic" (lokale Modelle), "advanced" (bessere lokale), "premium" (API-Modelle)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_tier = model_tier

        # RTX 2070 Spezifikation
        self.gpu_memory_gb = 8 if torch.cuda.is_available() else 0
        self.is_rtx2070 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory >= 7.5 * 1024**3
        )  # ~8GB

        # API-Clients für Premium-Modelle
        self.openai_client = None
        self.anthropic_client = None

        # Text-Modelle
        self.text_model = None
        self.text_tokenizer = None
        self.text_model_type = "local"  # "local", "openai", "anthropic"

        # Bild-Modelle
        self.vision_model = None
        self.vision_processor = None
        self.vision_model_type = "clip"  # "clip", "siglip"

        # Audio-Modelle
        self.audio_model = None
        self.audio_processor = None
        self.audio_model_type = "whisper"  # "whisper_base", "whisper_tiny"

        # Video-Modelle (für zukünftige Erweiterung)
        self.video_model = None

        self.load_models()

    def load_models(self):
        """Lädt alle multimodalen Modelle basierend auf dem gewählten Tier"""
        try:
            logger.info(f"🚀 Lade multimodale Modelle (Tier: {self.model_tier})...")

            if self.model_tier == "rtx2070":
                self._load_rtx2070_models()
            elif self.model_tier == "premium":
                self._load_premium_models()
            elif self.model_tier == "advanced":
                self._load_advanced_models()
            else:  # basic
                self._load_basic_models()

            logger.info("✅ Multimodale Modelle erfolgreich geladen")

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden multimodaler Modelle: {e}")
            # Fallback zu Basic-Modellen
            logger.info("🔄 Fallback zu Basic-Modellen...")
            self.model_tier = "basic"
            self._load_basic_models()

    def _load_premium_models(self):
        """Lädt Premium-API-Modelle (GPT-4, Claude)"""
        # OpenAI GPT-4
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
            self.text_model_type = "openai"
            logger.info("✅ OpenAI GPT-4 verfügbar")
        else:
            logger.warning("⚠️ OPENAI_API_KEY nicht gefunden, verwende lokale Modelle")

        # Anthropic Claude
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            logger.info("✅ Anthropic Claude verfügbar")
        else:
            logger.warning("⚠️ ANTHROPIC_API_KEY nicht gefunden")

        # Fortgeschrittene lokale Modelle als Fallback
        self._load_advanced_models()

    def _load_rtx2070_models(self):
        """Lädt RTX 2070 optimierte Modelle (8GB VRAM)"""
        logger.info("🎮 RTX 2070 Modus aktiviert - Optimiere für 8GB VRAM")

        # RTX 2070: 8-bit quantization für alle Modelle
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["lm_head"],  # Preserve output layer precision
            )
            if torch.cuda.is_available()
            else None
        )

        # Text-Modell: GPT-2 Medium (355M Parameter - passt in 8GB)
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "gpt2-medium",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            self.text_model_type = "gpt2_medium_rtx2070"
            logger.info("✅ GPT-2 Medium (RTX 2070 optimiert)")

        except Exception as e:
            logger.warning(f"⚠️ GPT-2 Medium fehlgeschlagen: {e}")
            # 4-bit Fallback
            quantization_4bit = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                if torch.cuda.is_available()
                else None
            )

            self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                quantization_config=quantization_4bit,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            self.text_model_type = "gpt2_base_rtx2070"
            logger.info("✅ GPT-2 Base (4-bit RTX 2070 Fallback)")

        # Vision-Modell: SigLIP Base (optimiert für 8GB)
        try:
            self.vision_processor = SiglipProcessor.from_pretrained(
                "google/siglip-base-patch16-224"
            )
            self.vision_model = SiglipModel.from_pretrained(
                "google/siglip-base-patch16-224",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.vision_model_type = "siglip_base_rtx2070"
            logger.info("✅ SigLIP Base (RTX 2070 optimiert)")

        except Exception as e:
            logger.warning(f"⚠️ SigLIP fehlgeschlagen: {e}")
            # CLIP Base Fallback
            self.vision_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_model_type = "clip_base_rtx2070"
            logger.info("✅ CLIP Base (RTX 2070 Fallback)")

        # Audio-Modell: Whisper Base (optimiert für 8GB)
        try:
            self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.audio_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-base",
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.audio_model_type = "whisper_base_rtx2070"
            logger.info("✅ Whisper Base (RTX 2070 optimiert)")

        except Exception as e:
            logger.warning(f"⚠️ Whisper Base fehlgeschlagen: {e}")
            # Tiny Fallback ohne quantization
            self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.audio_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-tiny",
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.audio_model_type = "whisper_tiny_rtx2070"
            logger.info("✅ Whisper Tiny (RTX 2070 Fallback)")

        # Speicher-Management für RTX 2070
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(
                f"🧹 GPU Speicher bereinigt - Verwendet: {torch.cuda.memory_allocated() // 1024**2}MB"
            )

    def _load_advanced_models(self):
        """Lädt fortgeschrittene lokale Modelle optimiert für RTX 2070 (8GB VRAM)"""
        # RTX 2070 Optimierung: 8-bit quantization für bessere Speichereffizienz
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # CPU offloading für große Modelle
            )
            if torch.cuda.is_available()
            else None
        )

        # Text-Modell (RTX 2070 optimiert: GPT-2 Medium statt XL)
        try:
            # RTX 2070: Verwende Medium statt XL für 8GB VRAM
            model_name = "gpt2-medium"  # 355M Parameter statt 1.5B

            self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

            # GPU-Memory-Management für RTX 2070
            if torch.cuda.is_available():
                # Speicher freigeben nach Laden
                torch.cuda.empty_cache()
                gc.collect()

            self.text_model_type = "local_advanced_rtx2070"
            logger.info("✅ GPT-2 Medium geladen (RTX 2070 optimiert - 8-bit)")

        except Exception as e:
            logger.warning(f"⚠️ GPT-2 Medium Laden fehlgeschlagen: {e}")
            # Fallback zu GPT-2 Base mit 4-bit quantization
            try:
                quantization_config_4bit = (
                    BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    if torch.cuda.is_available()
                    else None
                )

                self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    "gpt2",
                    quantization_config=quantization_config_4bit,
                    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                self.text_model_type = "local_basic_rtx2070"
                logger.info("✅ GPT-2 Base geladen (RTX 2070 - 4-bit Fallback)")

            except Exception as e2:
                logger.error(f"❌ Auch GPT-2 Base fehlgeschlagen: {e2}")
                # CPU-Only Fallback
                self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.text_model = AutoModelForCausalLM.from_pretrained("gpt2")
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
                self.text_model.to("cpu")
                self.text_model_type = "local_cpu_only"
                logger.info("✅ GPT-2 CPU-Only Fallback")

        # Vision-Modelle (RTX 2070 optimiert)
        try:
            # RTX 2070: Verwende kleinere SigLIP Variante
            vision_model_name = "google/siglip-base-patch16-224"  # Base statt Large

            self.vision_processor = SiglipProcessor.from_pretrained(vision_model_name)
            self.vision_model = SiglipModel.from_pretrained(
                vision_model_name,
                quantization_config=quantization_config,
                torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.vision_model_type = "siglip_rtx2070"
            logger.info("✅ SigLIP Base geladen (RTX 2070 optimiert)")

        except Exception as e:
            logger.warning(f"⚠️ SigLIP Laden fehlgeschlagen: {e}")
            # Fallback zu CLIP Base mit 4-bit
            try:
                quantization_config_4bit = (
                    BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    if torch.cuda.is_available()
                    else None
                )

                self.vision_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    quantization_config=quantization_config_4bit,
                    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                self.vision_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                self.vision_model_type = "clip_base_rtx2070"
                logger.info("✅ CLIP Base geladen (RTX 2070 - 4-bit Fallback)")

            except Exception as e2:
                logger.error(f"❌ Auch CLIP Base fehlgeschlagen: {e2}")
                # CPU-Only Fallback
                self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.vision_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self.vision_model.to("cpu")
                self.vision_model_type = "clip_cpu_only"
                logger.info("✅ CLIP CPU-Only Fallback")

        # Audio-Modelle (RTX 2070 optimiert)
        try:
            # RTX 2070: Verwende Whisper Base statt Large für 8GB VRAM
            audio_model_name = "openai/whisper-base"  # Base statt Large

            self.audio_processor = WhisperProcessor.from_pretrained(audio_model_name)
            self.audio_model = WhisperForConditionalGeneration.from_pretrained(
                audio_model_name,
                quantization_config=quantization_config,
                torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.audio_model_type = "whisper_base_rtx2070"
            logger.info("✅ Whisper Base geladen (RTX 2070 optimiert)")

        except Exception as e:
            logger.warning(f"⚠️ Whisper Base Laden fehlgeschlagen: {e}")
            # Fallback zu Whisper Tiny mit 4-bit
            try:
                quantization_config_4bit = (
                    BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    if torch.cuda.is_available()
                    else None
                )

                self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                self.audio_model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-tiny",
                    quantization_config=quantization_config_4bit,
                    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                self.audio_model_type = "whisper_tiny_rtx2070"
                logger.info("✅ Whisper Tiny geladen (RTX 2070 - 4-bit Fallback)")

            except Exception as e2:
                logger.error(f"❌ Auch Whisper Tiny fehlgeschlagen: {e2}")
                # CPU-Only Fallback
                self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                self.audio_model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-tiny"
                )
                self.audio_model.to("cpu")
                self.audio_model_type = "whisper_cpu_only"
                logger.info("✅ Whisper CPU-Only Fallback")

    def _load_basic_models(self):
        """Lädt grundlegende lokale Modelle (wie bisher)"""
        # Text-Modell (GPT-2 Large)
        self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.text_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_model.to(self.device)
        self.text_model_type = "local"

        # CLIP für Bildverständnis
        self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model.to(self.device)
        self.vision_model_type = "clip"

        # Whisper für Audio-Transkription
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.audio_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.audio_model.to(self.device)

        logger.info("✅ Basic-Modelle geladen")

    def process_text(self, text: str, max_length: int = 200) -> str:
        """Verarbeitet Text-Eingaben mit verschiedenen Modell-Typen"""
        try:
            if self.text_model_type == "openai" and self.openai_client:
                return self._process_text_openai(text, max_length)
            elif self.text_model_type == "anthropic" and self.anthropic_client:
                return self._process_text_anthropic(text, max_length)
            else:
                return self._process_text_local(text, max_length)

        except Exception as e:
            logger.error(f"Text-Verarbeitung fehlgeschlagen: {e}")
            return "Entschuldigung, ich konnte Ihre Texteingabe nicht verarbeiten."

    def _process_text_openai(self, text: str, max_length: int = 200) -> str:
        """Verarbeitet Text mit OpenAI GPT-4"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist eine fortschrittliche KI-Assistentin für politische Kommunikation in Deutschland. Antworte professionell, sachlich und hilfreich.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=max_length,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API Fehler: {e}")
            return self._process_text_local(text, max_length)  # Fallback

    def _process_text_anthropic(self, text: str, max_length: int = 200) -> str:
        """Verarbeitet Text mit Anthropic Claude"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_length,
                system="Du bist eine fortschrittliche KI-Assistentin für politische Kommunikation in Deutschland. Antworte professionell, sachlich und hilfreich.",
                messages=[{"role": "user", "content": text}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API Fehler: {e}")
            return self._process_text_local(text, max_length)  # Fallback

    def _process_text_local(self, text: str, max_length: int = 200) -> str:
        """Verarbeitet Text mit lokalen Modellen (GPT-2)"""
        try:
            inputs = self.text_tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.text_model.generate(
                    **inputs,
                    max_length=max_length + len(inputs["input_ids"][0]),
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.text_tokenizer.eos_token_id,
                )

            response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Entferne die ursprüngliche Eingabe aus der Antwort
            if response.startswith(text):
                response = response[len(text) :].strip()
            return response

        except Exception as e:
            logger.error(f"Lokale Text-Verarbeitung fehlgeschlagen: {e}")
            return "Entschuldigung, ich konnte Ihre Texteingabe nicht verarbeiten."

    def process_image(self, image_path: str) -> Dict[str, Union[str, float]]:
        """Verarbeitet Bild-Eingaben mit CLIP oder SigLIP"""
        try:
            image = Image.open(image_path)

            # Vordefinierte Beschreibungen für verschiedene Themen
            descriptions = [
                "eine politische Versammlung",
                "ein Regierungsgebäude",
                "eine Rede des Bundeskanzlers",
                "eine Pressekonferenz",
                "ein historisches Dokument",
                "eine Deutschlandkarte",
                "eine Wirtschaftsstatistik",
                "eine Umweltinitiative",
                "eine technologische Innovation",
                "eine internationale Konferenz",
            ]

            if self.vision_model_type == "siglip":
                return self._process_image_siglip(image, descriptions)
            else:
                return self._process_image_clip(image, descriptions)

        except Exception as e:
            logger.error(f"Bild-Verarbeitung fehlgeschlagen: {e}")
            return {
                "description": "Bild konnte nicht analysiert werden",
                "confidence": 0.0,
            }

    def _process_image_clip(
        self, image: Image.Image, descriptions: List[str]
    ) -> Dict[str, Union[str, float]]:
        """Verarbeitet Bild mit CLIP-Modell"""
        inputs = self.vision_processor(
            text=descriptions, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        best_idx = torch.argmax(probs[0]).item()
        confidence = probs[0][best_idx].item()

        return {
            "description": descriptions[best_idx],
            "confidence": confidence,
            "all_descriptions": descriptions,
            "all_probabilities": probs[0].cpu().numpy(),
            "model": "CLIP",
        }

    def _process_image_siglip(
        self, image: Image.Image, descriptions: List[str]
    ) -> Dict[str, Union[str, float]]:
        """Verarbeitet Bild mit SigLIP-Modell (höhere Genauigkeit)"""
        # SigLIP verwendet eine andere API
        inputs = self.vision_processor(
            text=descriptions,
            images=[image] * len(descriptions),  # Wiederhole Bild für jeden Text
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            # SigLIP gibt logits_per_image und logits_per_text zurück
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        best_idx = torch.argmax(probs[0]).item()
        confidence = probs[0][best_idx].item()

        return {
            "description": descriptions[best_idx],
            "confidence": confidence,
            "all_descriptions": descriptions,
            "all_probabilities": probs[0].cpu().numpy(),
            "model": "SigLIP",
        }

    def process_audio(self, audio_path: str) -> str:
        """Verarbeitet Audio-Eingaben mit Whisper"""
        try:
            # Lade Audio
            audio, sr = librosa.load(audio_path, sr=16000)

            # Whisper erwartet 16kHz Audio
            inputs = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                predicted_ids = self.audio_model.generate(**inputs)

            transcription = self.audio_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            return transcription

        except Exception as e:
            logger.error(f"Audio-Verarbeitung fehlgeschlagen: {e}")
            return "Audio konnte nicht transkribiert werden."

    def multimodal_response(
        self, text: str = None, image_path: str = None, audio_path: str = None
    ) -> Dict[str, str]:
        """
        Erstellt eine multimodale Antwort basierend auf allen verfügbaren Eingaben
        """
        response = {
            "text_response": "",
            "image_analysis": "",
            "audio_transcription": "",
            "integrated_response": "",
        }

        # Verarbeite Text
        if text:
            response["text_response"] = self.process_text(text)

        # Verarbeite Bild
        if image_path:
            image_result = self.process_image(image_path)
            response["image_analysis"] = (
                f"Das Bild zeigt {image_result['description']} (Konfidenz: {image_result['confidence']:.2f})"
            )

        # Verarbeite Audio
        if audio_path:
            response["audio_transcription"] = self.process_audio(audio_path)

        # Erstelle integrierte Antwort
        integrated_parts = []
        if text:
            integrated_parts.append(
                f"Basierend auf Ihrer Frage '{text}': {response['text_response']}"
            )
        if image_path:
            integrated_parts.append(response["image_analysis"])
        if audio_path:
            integrated_parts.append(f"Audio-Inhalt: {response['audio_transcription']}")

        response["integrated_response"] = " ".join(integrated_parts)

        return response


class AdvancedReasoningEngine:
    """
    Erweiterte Reasoning-Engine mit logischem Denken und Wissensintegration
    """

    def __init__(self):
        self.knowledge_base = {
            "politik": {
                "bundeskanzler": "Der Bundeskanzler ist das Staatsoberhaupt der Bundesrepublik Deutschland",
                "bundestag": "Das Parlament der Bundesrepublik Deutschland",
                "grundgesetz": "Die Verfassung der Bundesrepublik Deutschland",
            },
            "wirtschaft": {
                "bruttoinlandsprodukt": "Das Bruttoinlandsprodukt misst die Wirtschaftsleistung",
                "inflation": "Der Anstieg der Verbraucherpreise über einen Zeitraum",
                "arbeitslosigkeit": "Der Anteil der erwerbsfähigen Bevölkerung ohne Arbeit",
            },
        }

        # QA Pipeline für präzise Antworten
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1,
        )

    def reason_about_topic(
        self, question: str, context: str = None
    ) -> Dict[str, Union[str, float]]:
        """
        Führt logisches Reasoning über ein Thema durch
        """
        try:
            # Extrahiere Schlüsselkonzepte
            key_concepts = self._extract_key_concepts(question)

            # Suche relevantes Wissen
            relevant_knowledge = self._find_relevant_knowledge(key_concepts)

            # Wenn Kontext verfügbar, verwende QA-Modell
            if context:
                qa_result = self.qa_pipeline(question=question, context=context)
                answer = qa_result["answer"]
                confidence = qa_result["score"]
            else:
                answer = self._generate_reasoned_answer(question, relevant_knowledge)
                confidence = 0.7  # Standard-Konfidenz für generierte Antworten

            return {
                "answer": answer,
                "confidence": confidence,
                "key_concepts": key_concepts,
                "relevant_knowledge": relevant_knowledge,
                "reasoning_type": "qa_based" if context else "knowledge_based",
            }

        except Exception as e:
            logger.error(f"Reasoning fehlgeschlagen: {e}")
            return {
                "answer": "Ich konnte keine fundierte Antwort auf Ihre Frage finden.",
                "confidence": 0.0,
                "key_concepts": [],
                "relevant_knowledge": {},
                "reasoning_type": "error",
            }

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extrahiert Schlüsselkonzepte aus dem Text"""
        # Einfache Schlüsselwort-Extraktion (kann durch NER erweitert werden)
        keywords = [
            "politik",
            "wirtschaft",
            "bundeskanzler",
            "regierung",
            "gesetz",
            "europa",
            "usa",
            "china",
            "klima",
            "digitalisierung",
        ]

        found_concepts = []
        text_lower = text.lower()

        for keyword in keywords:
            if keyword in text_lower:
                found_concepts.append(keyword)

        return found_concepts

    def _find_relevant_knowledge(self, concepts: List[str]) -> Dict[str, str]:
        """Findet relevantes Wissen basierend auf Konzepten"""
        relevant = {}

        for concept in concepts:
            for category, knowledge in self.knowledge_base.items():
                if concept in knowledge:
                    relevant[concept] = knowledge[concept]

        return relevant

    def _generate_reasoned_answer(self, question: str, knowledge: Dict[str, str]) -> str:
        """Generiert eine durchdachte Antwort basierend auf Wissen"""
        if not knowledge:
            return "Basierend auf meinem aktuellen Wissensstand kann ich diese Frage nicht beantworten."

        # Baue eine zusammenhängende Antwort auf
        answer_parts = []
        for concept, info in knowledge.items():
            answer_parts.append(f"Zu {concept}: {info}")

        return " ".join(answer_parts)


# Kompatibilitätsfunktionen für bestehende Codebasis
def create_multimodal_model(model_tier="rtx2070"):
    """Erstellt ein multimodales Modell (standardmäßig RTX 2070 optimiert)"""
    return MultimodalTransformerModel(model_tier=model_tier)


def create_reasoning_engine():
    """Erstellt eine Reasoning-Engine"""
    return AdvancedReasoningEngine()
