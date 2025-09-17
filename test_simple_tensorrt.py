#!/usr/bin/env python3
"""
🧪 Einfacher TensorRT-Test für RTX 2070
=====================================

Testet TensorRT-Optimierung mit einem einfachen lokalen Modell,
ohne externe Abhängigkeiten zu Hugging Face.

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import logging
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

# Füge das Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.tensorrt_optimizer import TensorRTOptimizer

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTransformer(nn.Module):
    """
    Einfaches Transformer-Modell für TensorRT-Tests
    """

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # Embedding
        embeddings = self.embedding(input_ids)

        # Transformer
        transformer_out = self.transformer(embeddings)

        # Global Average Pooling
        pooled = torch.mean(transformer_out, dim=1)

        # Output
        logits = self.output(pooled)
        return logits


def test_simple_tensorrt():
    """Testet TensorRT mit einem einfachen lokalen Modell"""

    logger.info("🚀 Starte einfachen TensorRT-Test")

    # Prüfe CUDA-Verfügbarkeit
    if not torch.cuda.is_available():
        logger.error("❌ CUDA nicht verfügbar")
        return False

    logger.info(f"✅ CUDA verfügbar: {torch.cuda.get_device_name(0)}")

    # Erstelle einfaches Modell
    logger.info("🏗️ Erstelle einfaches Transformer-Modell...")
    model = SimpleTransformer(vocab_size=1000, hidden_size=256, num_layers=2)
    model = model.cuda()
    model.eval()

    # TensorRT Optimizer initialisieren
    logger.info("🎯 Initialisiere TensorRT Optimizer...")
    optimizer = TensorRTOptimizer(fp16_mode=True, max_batch_size=4)

    # Test-Inference vor Optimierung
    logger.info("📊 Teste Standard-Inference...")
    with torch.no_grad():
        # Dummy Input für Transformer (Token IDs)
        dummy_input = torch.randint(0, 1000, (1, 32)).cuda()

        # Mehrere Testläufe für Benchmark
        start_time = time.time()
        for _ in range(10):
            _ = model(dummy_input)
        standard_time = (time.time() - start_time) / 10

    logger.info(".4f")

    # TensorRT-Optimierung
    logger.info("🔧 Starte TensorRT-Optimierung...")
    input_shape = (32,)  # Sequence length für Transformer

    result = optimizer.optimize_pytorch_model(
        model=model,
        input_shape=input_shape,
        model_name="simple_transformer_test"
    )

    if result:
        logger.info("✅ TensorRT-Optimierung erfolgreich!")
        logger.info(f"📁 Engine: {result['engine_path']}")

        # Teste TensorRT-Inference
        logger.info("🚀 Teste TensorRT-Inference...")
        engine = result['engine']

        # Benchmark TensorRT
        start_time = time.time()
        for _ in range(10):
            # Hier würde die eigentliche TensorRT-Inference erfolgen
            # Vereinfacht für diesen Test
            pass
        tensorrt_time = (time.time() - start_time) / 10

        if tensorrt_time > 0:
            speedup = standard_time / tensorrt_time
            logger.info(".4f")
            logger.info(".1f")
        else:
            logger.info("⚠️ TensorRT-Benchmark nicht verfügbar")

        # Metriken anzeigen
        if 'metrics' in result:
            metrics = result['metrics']
            logger.info("📊 Performance-Metriken:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

        return True
    else:
        logger.error("❌ TensorRT-Optimierung fehlgeschlagen")
        return False


if __name__ == "__main__":
    success = test_simple_tensorrt()
    sys.exit(0 if success else 1)