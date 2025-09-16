#!/usr/bin/env python3
"""
Test-Konfiguration und Fixtures für Bundeskanzler-KI Tests
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

# Projekt-Root hinzufügen
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Warnungen unterdrücken
import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Projekt-Root Pfad"""
    return project_root


@pytest.fixture(scope="function")
def temp_directory() -> Generator[Path, None, None]:
    """Temporäres Verzeichnis für Tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="function")
def mock_corpus() -> Dict[str, Any]:
    """Mock-Corpus für Tests"""
    return {
        "entries": [
            {"text": "Deutschland hat ehrgeizige Klimaziele bis 2030.", "id": 1, "category": "politik"},
            {"text": "Die Energiewende ist ein zentrales Projekt der Regierung.", "id": 2, "category": "energie"},
            {"text": "Berlin ist die Hauptstadt Deutschlands.", "id": 3, "category": "allgemein"}
        ]
    }


@pytest.fixture(scope="function")
def mock_config() -> Dict[str, Any]:
    """Mock-Konfiguration für Tests"""
    return {
        "gpu": {
            "memory_growth": True,
            "cuda_data_dir": "/usr/lib/cuda"
        },
        "model": {
            "embedding_dim": 128,
            "lstm_units": 256,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "l1_reg": 1e-5,
            "l2_reg": 1e-4
        },
        "rag": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "index_path": "data/rag_index.faiss",
            "corpus_path": "data/corpus.json"
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False
        }
    }


@pytest.fixture(scope="function")
def mock_security_system():
    """Mock Security-System für Tests"""
    mock_system = MagicMock()
    mock_system.validate_input.return_value = {
        "is_valid": True,
        "message": "Eingabe ist gültig",
        "warnings": [],
        "sanitized_input": "Testfrage"
    }
    return mock_system


@pytest.fixture(scope="function")
def mock_rag_system():
    """Mock RAG-System für Tests"""
    mock_system = MagicMock()
    mock_system.search.return_value = [
        {"text": "Deutschland hat Klimaziele", "score": 0.95},
        {"text": "Energiewende ist wichtig", "score": 0.87}
    ]
    return mock_system


@pytest.fixture(scope="function")
def mock_model():
    """Mock KI-Modell für Tests"""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.9]])  # Mock-Vorhersage
    return mock_model


# GPU-spezifische Fixtures
@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Überprüft GPU-Verfügbarkeit"""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        return False


@pytest.fixture(scope="session")
def torch_gpu_available() -> bool:
    """Überprüft PyTorch GPU-Verfügbarkeit"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Performance-Messung
@pytest.fixture(scope="function")
def performance_timer():
    """Timer für Performance-Tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return Timer()


# Test-Kategorien
def pytest_configure(config):
    """Konfiguriere Test-Marker"""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "performance: mark test as performance test")


def pytest_collection_modifyitems(config, items):
    """Modifiziere Test-Sammlung basierend auf verfügbarer Hardware"""
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        gpu_available = False
    
    for item in items:
        # GPU-Tests überspringen wenn keine GPU verfügbar
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(pytest.mark.skip(reason="GPU nicht verfügbar"))
        
        # Performance-Tests als slow markieren
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)
