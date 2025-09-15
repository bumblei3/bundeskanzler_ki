#!/usr/bin/env python3
"""
Kontinuierliches Lernsystem für die Bundeskanzler KI
Automatisches Training, Modell-Optimierung und Performance-Monitoring
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from queue import Queue
from typing import Dict, List, Optional, Tuple

import GPUtil
import numpy as np
import psutil
import tensorflow as tf
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class ContinuousLearningSystem:
    """
    System für kontinuierliches Lernen und Modell-Optimierung
    """

    def __init__(self, model_path: str = "./models"):
        self.model_path = model_path
        self.base_model_name = "gpt2-large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lern-Queues
        self.feedback_queue = Queue()
        self.performance_queue = Queue()

        # Modell-Versionierung
        self.current_version = self._get_current_version()
        self.model_versions = self._load_model_versions()

        # Training-Konfiguration
        self.training_config = {
            "learning_rate": 5e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_steps": 1000,
            "save_steps": 500,
            "evaluation_steps": 250,
            "warmup_steps": 100,
        }

        # Monitoring
        self.performance_metrics = {
            "response_quality": [],
            "response_time": [],
            "user_satisfaction": [],
            "model_accuracy": [],
        }

        # Starte Hintergrund-Threads
        self._start_background_threads()

    def _start_background_threads(self):
        """Startet Hintergrund-Threads für kontinuierliches Lernen"""
        # Feedback-Verarbeitung
        feedback_thread = threading.Thread(
            target=self._process_feedback_loop, daemon=True
        )
        feedback_thread.start()

        # Performance-Monitoring
        monitoring_thread = threading.Thread(
            target=self._performance_monitoring_loop, daemon=True
        )
        monitoring_thread.start()

        # Automatisches Training
        training_thread = threading.Thread(target=self._auto_training_loop, daemon=True)
        training_thread.start()

        logger.info("✅ Kontinuierliches Lernsystem gestartet")

    def add_user_feedback(self, interaction: Dict):
        """
        Fügt User-Feedback zur Lern-Queue hinzu

        Args:
            interaction: Dictionary mit Interaktion-Daten
        """
        self.feedback_queue.put(
            {"timestamp": datetime.now(), "type": "user_feedback", "data": interaction}
        )

    def record_performance_metric(self, metric_type: str, value: float):
        """
        Zeichnet Performance-Metriken auf

        Args:
            metric_type: Typ der Metrik
            value: Metrikwert
        """
        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(
                {"timestamp": datetime.now(), "value": value}
            )

            # Behalte nur die letzten 1000 Einträge
            if len(self.performance_metrics[metric_type]) > 1000:
                self.performance_metrics[metric_type] = self.performance_metrics[
                    metric_type
                ][-1000:]

    def _process_feedback_loop(self):
        """Verarbeitet kontinuierlich User-Feedback"""
        while True:
            try:
                feedback = self.feedback_queue.get(timeout=1)

                if feedback["type"] == "user_feedback":
                    self._process_user_feedback(feedback["data"])

                self.feedback_queue.task_done()

            except:
                continue

    def _process_user_feedback(self, feedback_data: Dict):
        """Verarbeitet einzelnes User-Feedback"""
        # Analysiere Feedback und extrahiere Lernsignale
        if "rating" in feedback_data and feedback_data["rating"] < 3:
            # Niedrige Bewertung - analysiere warum
            self._analyze_low_rating_feedback(feedback_data)

        elif "correction" in feedback_data:
            # Korrektur bereitgestellt - füge zum Training hinzu
            self._add_correction_to_training_data(feedback_data)

    def _analyze_low_rating_feedback(self, feedback: Dict):
        """Analysiert Feedback mit niedriger Bewertung"""
        question = feedback.get("question", "")
        response = feedback.get("response", "")
        rating = feedback.get("rating", 0)

        # Einfache Analyse (kann durch ML-Modell erweitert werden)
        issues = []

        if len(response) < 10:
            issues.append("response_too_short")

        if "nicht" in response.lower() or "weiß nicht" in response.lower():
            issues.append("uncertain_response")

        # Logge für spätere Analyse
        logger.info(f"Niedrige Bewertung ({rating}) für Frage: {question[:50]}...")
        logger.info(f"Erkannte Probleme: {issues}")

    def _add_correction_to_training_data(self, feedback: Dict):
        """Fügt Korrekturen zu den Training-Daten hinzu"""
        question = feedback.get("question", "")
        wrong_response = feedback.get("response", "")
        correct_response = feedback.get("correction", "")

        if correct_response:
            # Erstelle Training-Paar
            training_example = {
                "input": question,
                "output": correct_response,
                "source": "user_correction",
                "timestamp": datetime.now().isoformat(),
            }

            # Speichere für späteres Training
            self._save_training_example(training_example)

    def _performance_monitoring_loop(self):
        """Überwacht kontinuierlich die Performance"""
        while True:
            try:
                # Sammle System-Metriken
                system_metrics = self._collect_system_metrics()

                # Sammle Modell-Metriken
                model_metrics = self._collect_model_metrics()

                # Speichere Metriken
                self.performance_queue.put(
                    {
                        "timestamp": datetime.now(),
                        "system": system_metrics,
                        "model": model_metrics,
                    }
                )

                # Prüfe auf Performance-Probleme
                self._check_performance_thresholds(system_metrics, model_metrics)

                time.sleep(60)  # Alle 60 Sekunden

            except Exception as e:
                logger.error(f"Performance-Monitoring Fehler: {e}")
                time.sleep(60)

    def _collect_system_metrics(self) -> Dict:
        """Sammelt System-Performance-Metriken"""
        try:
            # CPU und Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # GPU (falls verfügbar)
            gpu_metrics = {}
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_metrics = {
                        "gpu_usage": gpu.load * 100,
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal,
                        "gpu_temperature": gpu.temperature,
                    }
            except:
                pass

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                **gpu_metrics,
            }

        except Exception as e:
            logger.error(f"System-Metriken Fehler: {e}")
            return {}

    def _collect_model_metrics(self) -> Dict:
        """Sammelt Modell-Performance-Metriken"""
        # Berechne Durchschnittswerte der letzten Metriken
        metrics = {}

        for metric_type, values in self.performance_metrics.items():
            if values:
                recent_values = [v["value"] for v in values[-100:]]  # Letzte 100 Werte
                if recent_values:
                    metrics[f"{metric_type}_avg"] = np.mean(recent_values)
                    metrics[f"{metric_type}_std"] = np.std(recent_values)

        return metrics

    def _check_performance_thresholds(self, system_metrics: Dict, model_metrics: Dict):
        """Prüft auf Performance-Probleme und triggert Aktionen"""
        alerts = []

        # CPU-Schwellenwert
        if system_metrics.get("cpu_usage", 0) > 90:
            alerts.append("High CPU usage detected")

        # Memory-Schwellenwert
        if system_metrics.get("memory_usage", 0) > 90:
            alerts.append("High memory usage detected")

        # GPU-Schwellenwert
        if system_metrics.get("gpu_usage", 0) > 95:
            alerts.append("High GPU usage detected")

        # Modell-Performance
        if model_metrics.get("response_time_avg", 0) > 5.0:  # 5 Sekunden
            alerts.append("Slow response times detected")

        if model_metrics.get("user_satisfaction_avg", 0) < 3.0:  # Unter 3/5 Sternen
            alerts.append("Low user satisfaction detected")

        # Trigger Aktionen bei Alerts
        for alert in alerts:
            logger.warning(f"🚨 Performance Alert: {alert}")
            self._handle_performance_alert(alert)

    def _handle_performance_alert(self, alert: str):
        """Behandelt Performance-Alerts"""
        if "High CPU usage" in alert:
            # Reduziere Batch-Size oder erhöhe Wartezeiten
            self.training_config["batch_size"] = max(
                1, self.training_config["batch_size"] // 2
            )

        elif "High memory usage" in alert:
            # Aktiviere Memory-Optimierung
            self._enable_memory_optimization()

        elif "Slow response times" in alert:
            # Optimiere Modell oder verwende Caching
            self._optimize_response_times()

    def _auto_training_loop(self):
        """Führt automatisches Training durch"""
        while True:
            try:
                # Prüfe, ob genug neue Daten für Training vorhanden sind
                if self._should_trigger_training():
                    logger.info("🚀 Starte automatisches Modell-Training...")

                    # Sammle Training-Daten
                    training_data = self._collect_training_data()

                    if training_data:
                        # Führe Training durch
                        self._perform_incremental_training(training_data)

                        # Erstelle neue Modell-Version
                        self._create_new_model_version()

                    logger.info("✅ Automatisches Training abgeschlossen")

                # Warte 24 Stunden bis zum nächsten Check
                time.sleep(24 * 60 * 60)

            except Exception as e:
                logger.error(f"Auto-Training Fehler: {e}")
                time.sleep(60 * 60)  # Warte 1 Stunde bei Fehlern

    def _should_trigger_training(self) -> bool:
        """Prüft, ob Training ausgelöst werden sollte"""
        # Kriterien für Training:
        # 1. Mindestens 100 neue Feedback-Einträge
        # 2. User-Zufriedenheit unter Schwellenwert
        # 3. Modell ist älter als 7 Tage

        feedback_count = self.feedback_queue.qsize()
        satisfaction_avg = np.mean(
            [m["value"] for m in self.performance_metrics["user_satisfaction"][-50:]]
            or [4.0]
        )

        last_training = self._get_last_training_time()
        days_since_training = (datetime.now() - last_training).days

        return (
            feedback_count >= 100 or satisfaction_avg < 3.5 or days_since_training >= 7
        )

    def _collect_training_data(self) -> List[Dict]:
        """Sammelt Training-Daten aus Feedback"""
        training_data = []

        # Sammle aus Feedback-Queue
        while not self.feedback_queue.empty():
            feedback = self.feedback_queue.get()
            if feedback["type"] == "user_feedback":
                data = feedback["data"]
                if "correction" in data:
                    training_data.append(
                        {
                            "input": data["question"],
                            "output": data["correction"],
                            "source": "user_feedback",
                        }
                    )

        return training_data

    def _perform_incremental_training(self, training_data: List[Dict]):
        """Führt inkrementelles Training durch"""
        try:
            # Konvertiere zu Dataset
            dataset = Dataset.from_list(training_data)

            # Tokenisiere Daten
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            tokenizer.pad_token = tokenizer.eos_token

            def tokenize_function(examples):
                inputs = tokenizer(
                    examples["input"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                )
                outputs = tokenizer(
                    examples["output"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                )
                inputs["labels"] = outputs["input_ids"]
                return inputs

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Lade aktuelles Modell
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(self.model_path, f"v{self.current_version}")
            ).to(self.device)

            # Training-Argumente
            training_args = TrainingArguments(
                output_dir=os.path.join(self.model_path, "checkpoints"),
                num_train_epochs=1,
                per_device_train_batch_size=self.training_config["batch_size"],
                gradient_accumulation_steps=self.training_config[
                    "gradient_accumulation_steps"
                ],
                learning_rate=self.training_config["learning_rate"],
                warmup_steps=self.training_config["warmup_steps"],
                save_steps=self.training_config["save_steps"],
                evaluation_steps=self.training_config["evaluation_steps"],
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                ),
            )

            # Training
            trainer.train()

            # Speichere Modell
            new_version = self.current_version + 1
            trainer.save_model(os.path.join(self.model_path, f"v{new_version}"))

            logger.info(f"✅ Modell-Version v{new_version} gespeichert")

        except Exception as e:
            logger.error(f"Inkrementelles Training fehlgeschlagen: {e}")

    def _create_new_model_version(self):
        """Erstellt eine neue Modell-Version"""
        self.current_version += 1
        self.model_versions[self.current_version] = {
            "created": datetime.now().isoformat(),
            "training_data_size": len(self._collect_training_data()),
            "base_version": self.current_version - 1,
        }

        self._save_model_versions()

    def _get_current_version(self) -> int:
        """Ermittelt die aktuelle Modell-Version"""
        if os.path.exists(os.path.join(self.model_path, "versions.json")):
            with open(os.path.join(self.model_path, "versions.json"), "r") as f:
                versions = json.load(f)
                return max(versions.keys()) if versions else 1
        return 1

    def _load_model_versions(self) -> Dict:
        """Lädt Modell-Versionen"""
        versions_file = os.path.join(self.model_path, "versions.json")
        if os.path.exists(versions_file):
            with open(versions_file, "r") as f:
                return json.load(f)
        return {
            1: {
                "created": datetime.now().isoformat(),
                "training_data_size": 0,
                "base_version": None,
            }
        }

    def _save_model_versions(self):
        """Speichert Modell-Versionen"""
        versions_file = os.path.join(self.model_path, "versions.json")
        with open(versions_file, "w") as f:
            json.dump(self.model_versions, f, indent=2)

    def _get_last_training_time(self) -> datetime:
        """Ermittelt den Zeitpunkt des letzten Trainings"""
        if self.model_versions:
            latest_version = max(self.model_versions.keys())
            created_str = self.model_versions[latest_version]["created"]
            return datetime.fromisoformat(created_str)
        return datetime.now() - timedelta(days=30)  # Fallback: vor 30 Tagen

    def _save_training_example(self, example: Dict):
        """Speichert ein Training-Beispiel"""
        training_file = os.path.join(self.model_path, "training_data.jsonl")

        with open(training_file, "a", encoding="utf-8") as f:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

    def _enable_memory_optimization(self):
        """Aktiviert Memory-Optimierung"""
        logger.info("🔧 Aktiviere Memory-Optimierung...")
        # Implementierung von Memory-Optimierungen
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _optimize_response_times(self):
        """Optimiert Antwortzeiten"""
        logger.info("⚡ Optimiere Antwortzeiten...")
        # Implementierung von Response-Time-Optimierungen
        # z.B. Modell-Quantisierung, Caching, etc.


# Globale Instanz für kontinuierliches Lernen
continuous_learning_system = ContinuousLearningSystem()


def get_continuous_learning_system():
    """Gibt die globale Instanz des kontinuierlichen Lernsystems zurück"""
    return continuous_learning_system
