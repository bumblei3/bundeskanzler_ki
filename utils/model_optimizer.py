"""
Modell-Optimierung für Bundeskanzler-KI
Implementiert Quantisierung, Pruning, Knowledge Distillation und Modell-Kompression
"""

import tensorflow as tf
import numpy as np
import os
import logging
from typing import Optional, Dict, Any, Tuple, List
import tempfile
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Umfassende Modell-Optimierungsklasse"""

    def __init__(self, model_path: str = "bundeskanzler_ki_model.keras"):
        self.model_path = model_path
        self.original_model = None
        self.optimized_models = {}

    def load_model(self) -> tf.keras.Model:
        """Lädt das Originalmodell"""
        try:
            # GPU-Probleme vermeiden beim Laden
            with tf.device('/CPU:0'):
                self.original_model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Modell geladen aus {self.model_path}")
            return self.original_model
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            raise

    def quantize_model_int8(self, model: tf.keras.Model) -> tf.keras.Model:
        """Quantisiert Modell zu int8 für bessere Performance"""
        logger.info("🔄 Starte int8-Quantisierung...")

        try:
            # TensorFlow Lite Converter für Quantisierung
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Optimierungen aktivieren
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]

            # Wichtige Fixes für GRU-Kompatibilität
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False

            # Representative Dataset für Kalibrierung (vereinfacht)
            def representative_dataset():
                # Erstelle repräsentative Daten für Quantisierung
                for _ in range(100):
                    data = np.random.randint(0, 100, size=(1, 100), dtype=np.int32)
                    yield [data.astype(np.float32)]

            converter.representative_dataset = representative_dataset

            # Konvertierung
            quantized_tflite_model = converter.convert()

            # Speichere quantisiertes Modell
            quantized_path = self.model_path.replace('.keras', '_quantized_int8.tflite')
            with open(quantized_path, 'wb') as f:
                f.write(quantized_tflite_model)

            logger.info(f"✅ int8-Quantisiertes Modell gespeichert: {quantized_path}")

            # Interpreter für Inference
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
            interpreter.allocate_tensors()

            self.optimized_models['int8'] = {
                'path': quantized_path,
                'interpreter': interpreter,
                'type': 'tflite_int8'
            }

            return interpreter

        except Exception as e:
            logger.warning(f"int8-Quantisierung fehlgeschlagen: {e}")
            logger.info("💡 Verwende alternative Optimierungen...")
            return None

    def quantize_model_float16(self, model: tf.keras.Model) -> tf.keras.Model:
        """Quantisiert Modell zu float16 für GPU-Performance"""
        logger.info("🔄 Starte float16-Quantisierung...")

        try:
            # Float16 Quantisierung
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            # GRU-Kompatibilität
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False

            quantized_tflite_model = converter.convert()

            # Speichere quantisiertes Modell
            quantized_path = self.model_path.replace('.keras', '_quantized_fp16.tflite')
            with open(quantized_path, 'wb') as f:
                f.write(quantized_tflite_model)

            logger.info(f"✅ float16-Quantisiertes Modell gespeichert: {quantized_path}")

            # Interpreter für Inference
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
            interpreter.allocate_tensors()

            self.optimized_models['fp16'] = {
                'path': quantized_path,
                'interpreter': interpreter,
                'type': 'tflite_fp16'
            }

            return interpreter

        except Exception as e:
            logger.warning(f"float16-Quantisierung fehlgeschlagen: {e}")
            logger.info("💡 Verwende alternative Optimierungen...")
            return None

    def apply_pruning(self, model: tf.keras.Model, pruning_rate: float = 0.2) -> tf.keras.Model:
        """Wendet Pruning auf das Modell an"""
        logger.info(f"🔄 Wende Pruning mit Rate {pruning_rate} an...")

        try:
            import tensorflow_model_optimization as tfmot

            # Pruning Schedule
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

            # Pruning konfigurieren
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=pruning_rate,
                    begin_step=0,
                    end_step=1000
                )
            }

            # Pruning auf alle Dense und GRU Layer anwenden
            def apply_pruning_to_layer(layer):
                if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.GRU)):
                    return prune_low_magnitude(layer, **pruning_params)
                return layer

            # Modell mit Pruning neu erstellen
            pruned_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_layer
            )

            # Kompilierung
            pruned_model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            logger.info("✅ Pruning erfolgreich angewendet")
            self.optimized_models['pruned'] = {
                'model': pruned_model,
                'pruning_rate': pruning_rate,
                'type': 'pruned'
            }

            return pruned_model

        except ImportError:
            logger.warning("TensorFlow Model Optimization nicht verfügbar, überspringe Pruning")
            logger.info("💡 Installiere mit: pip install tensorflow-model-optimization")
            return model

    def apply_knowledge_distillation(self, teacher_model: tf.keras.Model,
                                   student_model: tf.keras.Model,
                                   x_train: np.ndarray, y_train: np.ndarray,
                                   temperature: float = 3.0, alpha: float = 0.1) -> tf.keras.Model:
        """Wendet Knowledge Distillation an"""
        logger.info("🔄 Starte Knowledge Distillation...")

        class DistillationLoss(tf.keras.losses.Loss):
            def __init__(self, temperature, alpha):
                super().__init__()
                self.temperature = temperature
                self.alpha = alpha

            def call(self, y_true, y_pred):
                # Student Loss
                y_true_soft = tf.nn.softmax(y_true / self.temperature)
                y_pred_soft = tf.nn.softmax(y_pred / self.temperature)

                # Knowledge Distillation Loss
                kd_loss = tf.keras.losses.categorical_crossentropy(
                    y_true_soft, y_pred_soft, from_logits=False
                ) * (self.temperature ** 2)

                # Student Loss
                student_loss = tf.keras.losses.categorical_crossentropy(
                    y_true, y_pred, from_logits=True
                )

                # Kombinierte Loss
                return self.alpha * kd_loss + (1 - self.alpha) * student_loss

        # Teacher Model für Soft Targets
        teacher_logits = teacher_model.predict(x_train, verbose=0)

        # Student Model kompilieren mit Distillation Loss
        student_model.compile(
            loss=DistillationLoss(temperature, alpha),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        # Training mit Knowledge Distillation
        student_model.fit(
            x_train, teacher_logits,
            epochs=10,
            batch_size=32,
            verbose=1
        )

        logger.info("✅ Knowledge Distillation abgeschlossen")
        self.optimized_models['distilled'] = {
            'model': student_model,
            'temperature': temperature,
            'alpha': alpha,
            'type': 'distilled'
        }

        return student_model

    def create_compressed_model(self, model: tf.keras.Model,
                              compression_factor: float = 0.5) -> tf.keras.Model:
        """Erstellt ein komprimiertes Modell mit reduzierter Komplexität"""
        logger.info(f"🔄 Komprimiere Modell um Faktor {compression_factor}...")

        # Analysiere aktuelle Modell-Größe
        original_params = model.count_params()

        # Reduziere Layer-Größen
        def compress_layer(layer):
            if isinstance(layer, tf.keras.layers.Embedding):
                # Embedding Layer komprimieren
                new_units = max(32, int(layer.output_dim * compression_factor))
                return tf.keras.layers.Embedding(
                    input_dim=layer.input_dim,
                    output_dim=new_units,
                    input_length=layer.input_length
                )
            elif isinstance(layer, tf.keras.layers.GRU):
                # GRU Layer komprimieren
                new_units = max(32, int(layer.units * compression_factor))
                return tf.keras.layers.GRU(
                    new_units,
                    dropout=layer.dropout,
                    recurrent_dropout=layer.recurrent_dropout,
                    return_sequences=layer.return_sequences
                )
            elif isinstance(layer, tf.keras.layers.Dense):
                # Dense Layer komprimieren
                new_units = max(32, int(layer.units * compression_factor))
                return tf.keras.layers.Dense(
                    new_units,
                    activation=layer.activation
                )
            return layer

        # Neues komprimiertes Modell erstellen
        compressed_model = tf.keras.models.clone_model(
            model,
            clone_function=compress_layer
        )

        # Kompilierung
        compressed_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        compressed_params = compressed_model.count_params()
        compression_ratio = compressed_params / original_params

        logger.info(f"✅ Modell komprimiert: {original_params} → {compressed_params} Parameter ({compression_ratio:.2f}x)")

        self.optimized_models['compressed'] = {
            'model': compressed_model,
            'compression_factor': compression_factor,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'type': 'compressed'
        }

        return compressed_model

    def benchmark_models(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Benchmarkt alle optimierten Modelle"""
        logger.info("🚀 Starte Modell-Benchmarking...")

        results = {}
        batch_size = 32

        # Original Modell benchmarken
        if self.original_model:
            logger.info("📊 Benchmarke Original-Modell...")
            start_time = time.time()
            predictions = self.original_model.predict(test_data, batch_size=batch_size, verbose=0)
            inference_time = time.time() - start_time

            loss, accuracy = self.original_model.evaluate(test_data, test_labels, verbose=0)

            results['original'] = {
                'inference_time': inference_time,
                'accuracy': accuracy,
                'loss': loss,
                'model_size': self._get_model_size(self.original_model),
                'parameters': self.original_model.count_params()
            }

        # Optimierte Modelle benchmarken
        for name, model_info in self.optimized_models.items():
            logger.info(f"📊 Benchmarke {name}-Modell...")

            try:
                if 'interpreter' in model_info:
                    # TFLite Modell
                    interpreter = model_info['interpreter']
                    start_time = time.time()

                    # Inference für alle Test-Daten
                    for i in range(0, len(test_data), batch_size):
                        batch = test_data[i:i+batch_size]
                        # TFLite Inference (vereinfacht)
                        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                                             batch.astype(np.float32))
                        interpreter.invoke()
                        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

                    inference_time = time.time() - start_time

                    results[name] = {
                        'inference_time': inference_time,
                        'model_size': os.path.getsize(model_info['path']) if 'path' in model_info else 0,
                        'type': model_info['type']
                    }

                elif 'model' in model_info:
                    # Keras Modell
                    model = model_info['model']
                    start_time = time.time()
                    predictions = model.predict(test_data, batch_size=batch_size, verbose=0)
                    inference_time = time.time() - start_time

                    results[name] = {
                        'inference_time': inference_time,
                        'model_size': self._get_model_size(model),
                        'parameters': model.count_params(),
                        'type': model_info['type']
                    }

            except Exception as e:
                logger.error(f"Fehler beim Benchmarking von {name}: {e}")
                results[name] = {'error': str(e)}

        return results

    def _get_model_size(self, model: tf.keras.Model) -> int:
        """Gibt die Größe des Modells in Bytes zurück"""
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            model.save(f.name)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        return size

    def save_optimization_report(self, benchmark_results: Dict, output_path: str = "model_optimization_report.json"):
        """Speichert einen Optimierungsbericht"""
        report = {
            'timestamp': time.time(),
            'original_model': self.model_path,
            'optimizations_applied': list(self.optimized_models.keys()),
            'benchmark_results': benchmark_results,
            'recommendations': self._generate_recommendations(benchmark_results)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Optimierungsbericht gespeichert: {output_path}")
        return report

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generiert Empfehlungen basierend auf Benchmark-Ergebnissen"""
        recommendations = []

        if 'original' in results and len(results) > 1:
            original_time = results['original'].get('inference_time', float('inf'))

            # Finde schnellstes Modell
            fastest_model = min(
                [(name, data.get('inference_time', float('inf')))
                 for name, data in results.items() if name != 'original'],
                key=lambda x: x[1],
                default=None
            )

            if fastest_model and fastest_model[1] < original_time:
                speedup = original_time / fastest_model[1]
                recommendations.append(
                    f"Verwende {fastest_model[0]}-Modell für {speedup:.1f}x schnellere Inference"
                )

        # Größenvergleich
        if len(results) > 1:
            sizes = [(name, data.get('model_size', 0)) for name, data in results.items()]
            smallest = min(sizes, key=lambda x: x[1])
            largest = max(sizes, key=lambda x: x[1])

            if smallest[1] > 0 and largest[1] > 0:
                ratio = largest[1] / smallest[1]
                recommendations.append(
                    f"{smallest[0]}-Modell ist {ratio:.1f}x kleiner als {largest[0]}-Modell"
                )

        return recommendations

    def optimize_all(self, x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Führt alle verfügbaren Optimierungen durch und erstellt Bericht"""
        logger.info("🚀 Starte vollständige Modell-Optimierung...")

        # Original Modell laden
        model = self.load_model()

        # Verfügbare Optimierungen anwenden
        try:
            # Quantisierung (funktioniert möglicherweise nicht mit GRU)
            int8_result = self.quantize_model_int8(model)
            fp16_result = self.quantize_model_float16(model)

            # Pruning (falls verfügbar)
            pruned_model = self.apply_pruning(model)

            # Kompression (immer verfügbar)
            compressed_model = self.create_compressed_model(model)

            # Knowledge Distillation (immer verfügbar)
            distilled_model = self.apply_knowledge_distillation(
                model, compressed_model, x_train, y_train
            )

        except Exception as e:
            logger.error(f"Fehler bei Optimierungen: {e}")

        # Benchmarking
        benchmark_results = self.benchmark_models(x_test, y_test)

        # Bericht erstellen
        report = self.save_optimization_report(benchmark_results)

        logger.info("✅ Modell-Optimierung abgeschlossen")
        return report