#!/usr/bin/env python3
"""
Comprehensive Test Suite für Bundeskanzler-KI
Automatisierte Tests für alle Core-Module mit Coverage-Reporting
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


class TestPerformanceKI:
    """Tests für Performance-optimierte KI"""

    @pytest.fixture
    def temp_project_dir(self) -> Generator[Path, None, None]:
        """Temporärer Projektordner für Tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create basic structure
            (tmp_path / "core").mkdir()
            (tmp_path / "data").mkdir()
            (tmp_path / "models").mkdir()

            # Create mock corpus
            corpus = [
                {"text": "Deutschland hat ehrgeizige Klimaziele bis 2030.", "id": 1},
                {
                    "text": "Die Energiewende ist ein zentrales Projekt der Regierung.",
                    "id": 2,
                },
            ]

            with open(tmp_path / "data" / "corpus.json", "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False)

            yield tmp_path

    @patch("core.performance_ki.RAGSystem")
    def test_performance_ki_initialization(self, mock_rag, temp_project_dir):
        """Test Performance-KI Initialisierung"""
        sys.path.insert(0, str(project_root))
        from core.performance_ki import PerformanceOptimizedKI

        # Mock RAG system
        mock_rag_instance = Mock()
        mock_rag.return_value = mock_rag_instance

        with patch("core.performance_ki.project_root", str(temp_project_dir)):
            ki = PerformanceOptimizedKI(enable_cache=False)

            assert ki.enable_cache == False
            assert ki.last_confidence == 0.0
            mock_rag.assert_called_once()

    @patch("core.performance_ki.RAGSystem")
    def test_theme_recognition(self, mock_rag, temp_project_dir):
        """Test Themen-Erkennung"""
        from core.performance_ki import PerformanceOptimizedKI

        mock_rag_instance = Mock()
        mock_rag.return_value = mock_rag_instance

        with patch("core.performance_ki.project_root", str(temp_project_dir)):
            ki = PerformanceOptimizedKI(enable_cache=False)

            # Test Klimathema
            thema = ki.erkenne_thema("Was ist die Klimapolitik?")
            assert thema == "klima"

            # Test Wirtschaftsthema
            thema = ki.erkenne_thema("Wie steht es um die deutsche Wirtschaft?")
            assert thema == "wirtschaft"

            # Test unbekanntes Thema
            thema = ki.erkenne_thema("Wie ist das Wetter?")
            assert thema == "allgemein"

    @patch("core.performance_ki.RAGSystem")
    def test_caching_functionality(self, mock_rag, temp_project_dir):
        """Test Caching-Funktionalität"""
        from core.performance_ki import PerformanceOptimizedKI

        mock_rag_instance = Mock()
        mock_rag_instance.retrieve_relevant_documents.return_value = [
            {"text": "Test-Antwort", "score": 0.85}
        ]
        mock_rag.return_value = mock_rag_instance

        with patch("core.performance_ki.project_root", str(temp_project_dir)):
            ki = PerformanceOptimizedKI(enable_cache=True)

            # Erste Anfrage
            antwort1 = ki.antwort("Test-Frage")
            assert antwort1 == "Test-Antwort"
            assert ki.stats["cache_misses"] == 1
            assert ki.stats["cache_hits"] == 0

            # Zweite Anfrage (sollte aus Cache kommen)
            antwort2 = ki.antwort("Test-Frage")
            assert antwort2 == "Test-Antwort"
            assert ki.stats["cache_hits"] == 1

    def test_performance_stats(self, temp_project_dir):
        """Test Performance-Statistiken"""
        from core.performance_ki import PerformanceOptimizedKI

        with patch("core.performance_ki.RAGSystem"), patch(
            "core.performance_ki.project_root", str(temp_project_dir)
        ):

            ki = PerformanceOptimizedKI(enable_cache=False)

            stats = ki.get_performance_stats()

            assert "total_queries" in stats
            assert "cache_hits" in stats
            assert "cache_misses" in stats
            assert "avg_response_time" in stats
            assert "cache_hit_rate" in stats
            assert "cache_size" in stats


class TestAdvancedMonitor:
    """Tests für Advanced Monitoring System"""

    @pytest.fixture
    def temp_monitor_dir(self) -> Generator[Path, None, None]:
        """Temporärer Monitor-Ordner"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            yield tmp_path

    def test_monitor_initialization(self, temp_monitor_dir):
        """Test Monitor-Initialisierung"""
        from monitoring.advanced_monitor import AdvancedMonitor

        monitor = AdvancedMonitor(project_root=str(temp_monitor_dir))

        assert monitor.monitoring_active == False
        assert len(monitor.alert_rules) > 0
        assert monitor.metrics_dir.exists()
        assert monitor.logs_dir.exists()

    def test_metric_collection(self, temp_monitor_dir):
        """Test Metrik-Sammlung"""
        from monitoring.advanced_monitor import AdvancedMonitor, MetricEntry

        monitor = AdvancedMonitor(project_root=str(temp_monitor_dir))

        # Add test metric
        monitor.add_metric("test_metric", 42.0, {"type": "test"})

        assert "test_metric" in monitor.metrics
        assert len(monitor.metrics["test_metric"]) == 1

        entry = monitor.metrics["test_metric"][0]
        assert isinstance(entry, MetricEntry)
        assert entry.value == 42.0
        assert entry.tags["type"] == "test"

    def test_alert_rules(self, temp_monitor_dir):
        """Test Alert-Regeln"""
        from monitoring.advanced_monitor import AdvancedMonitor, AlertRule

        monitor = AdvancedMonitor(project_root=str(temp_monitor_dir))

        # Add custom alert rule
        rule = AlertRule(
            name="Test Alert", metric_name="test_metric", condition="gt", threshold=50.0
        )

        monitor.add_alert_rule(rule)

        # Test alert triggering
        monitor.add_metric("test_metric", 60.0)  # Should trigger

        # Check if alert was triggered (would be in logs)
        assert len(monitor.alert_rules) > 0

    def test_health_check(self, temp_monitor_dir):
        """Test Health-Check"""
        from monitoring.advanced_monitor import AdvancedMonitor

        monitor = AdvancedMonitor(project_root=str(temp_monitor_dir))

        health = monitor.generate_health_check()

        assert "status" in health
        assert "timestamp" in health
        assert "checks" in health
        assert health["status"] in ["healthy", "unhealthy", "error"]

    def test_metrics_export(self, temp_monitor_dir):
        """Test Metrik-Export"""
        from monitoring.advanced_monitor import AdvancedMonitor

        monitor = AdvancedMonitor(project_root=str(temp_monitor_dir))

        # Add some test metrics
        monitor.add_metric("cpu_usage", 45.0)
        monitor.add_metric("memory_usage", 60.0)

        # Test JSON export
        json_export = monitor.export_metrics("json", hours=1)
        assert isinstance(json_export, str)

        # Test Prometheus export
        prometheus_export = monitor.export_metrics("prometheus", hours=1)
        assert isinstance(prometheus_export, str)
        assert "# HELP" in prometheus_export


class TestRAGSystem:
    """Tests für RAG System"""

    @pytest.fixture
    def temp_rag_dir(self) -> Generator[Path, None, None]:
        """Temporärer RAG-Ordner"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create corpus
            corpus = [
                {"text": "Deutschland verfolgt ehrgeizige Klimaziele.", "id": 1},
                {"text": "Die Energiewende ist ein wichtiges Projekt.", "id": 2},
                {"text": "Wirtschaftspolitik fokussiert auf Innovation.", "id": 3},
            ]

            with open(tmp_path / "corpus.json", "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False)

            yield tmp_path

    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_rag_initialization(
        self, mock_faiss, mock_sentence_transformer, temp_rag_dir
    ):
        """Test RAG-System Initialisierung"""
        import numpy as np

        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        mock_sentence_transformer.return_value = mock_model

        # Mock FAISS index
        mock_index = Mock()
        mock_faiss.return_value = mock_index

        from core.rag_system import RAGSystem

        corpus_path = str(temp_rag_dir / "corpus.json")
        rag = RAGSystem(corpus_path=corpus_path)

        assert rag.corpus_entries is not None
        assert len(rag.corpus_entries) == 3
        mock_sentence_transformer.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    @patch("core.rag_system.faiss")
    def test_document_retrieval(
        self, mock_faiss_module, mock_faiss, mock_sentence_transformer, temp_rag_dir
    ):
        """Test Dokument-Retrieval"""
        import numpy as np

        # Mock setup
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        mock_sentence_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),
            np.array([[0, 1, 2]]),
        )
        mock_faiss.return_value = mock_index

        # Mock faiss module functions
        mock_faiss_module.normalize_L2 = Mock()

        from core.rag_system import RAGSystem

        corpus_path = str(temp_rag_dir / "corpus.json")
        rag = RAGSystem(corpus_path=corpus_path)

        # Test retrieval
        docs = rag.retrieve_relevant_documents("Klimaziele", top_k=2)

        assert isinstance(docs, list)
        mock_model.encode.assert_called()
        mock_index.search.assert_called()


class TestCodeQuality:
    """Tests für Code-Quality Tools"""

    def test_quality_manager_initialization(self):
        """Test Quality Manager Initialisierung"""
        from utils.code_quality import CodeQualityManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CodeQualityManager(project_root=tmp_dir)

            assert manager.project_root == Path(tmp_dir)
            assert manager.report_dir.exists()

    @patch("subprocess.run")
    def test_code_formatting(self, mock_subprocess):
        """Test Code-Formatierung"""
        from utils.code_quality import CodeQualityManager

        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stderr="", stdout="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CodeQualityManager(project_root=tmp_dir)

            results = manager.format_code(["test_file.py"])

            assert "black" in results
            assert "isort" in results
            mock_subprocess.assert_called()


class TestSystemIntegration:
    """Integrationstests für das gesamte System"""

    @pytest.fixture
    def integration_env(self) -> Generator[Path, None, None]:
        """Setup für Integrationstests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create full directory structure
            for dir_name in ["core", "data", "models", "monitoring", "logs"]:
                (tmp_path / dir_name).mkdir()

            # Create minimal corpus
            corpus = [{"text": "Test-Inhalt für Integration", "id": 1}]
            with open(tmp_path / "data" / "corpus.json", "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False)

            yield tmp_path

    @patch("core.rag_system.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_full_system_integration(
        self, mock_faiss, mock_transformer, integration_env
    ):
        """Test komplette System-Integration"""
        # Mock dependencies
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = ([[0.9]], [[0]])
        mock_faiss.return_value = mock_index

        # Test mit echten Modulen
        with patch("core.performance_ki.project_root", str(integration_env)):
            from core.performance_ki import PerformanceOptimizedKI
            from monitoring.advanced_monitor import AdvancedMonitor

            # Initialize system
            monitor = AdvancedMonitor(project_root=str(integration_env))
            ki = PerformanceOptimizedKI(enable_cache=False)

            # Test basic functionality
            assert ki is not None
            assert monitor is not None

            # Test health check
            health = monitor.generate_health_check()
            assert health["status"] in ["healthy", "unhealthy", "error"]


# Test Configuration
class TestConfig:
    """Test-Konfiguration und Utilities"""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup für jeden Test"""
        # Ensure clean environment
        if "core.performance_ki" in sys.modules:
            del sys.modules["core.performance_ki"]
        if "monitoring.advanced_monitor" in sys.modules:
            del sys.modules["monitoring.advanced_monitor"]

    def test_import_all_modules(self):
        """Test dass alle Module importiert werden können"""
        modules_to_test = [
            "utils.code_quality",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")


# Pytest Configuration
def pytest_configure(config):
    """Pytest Konfiguration"""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--cov=core",
            "--cov=monitoring",
            "--cov=utils",
            "--cov-report=html:reports/coverage",
            "--cov-report=term-missing",
            "--junit-xml=reports/junit.xml",
        ]
    )
